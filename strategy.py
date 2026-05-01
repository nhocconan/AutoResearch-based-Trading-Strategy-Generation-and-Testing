#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ATR-based volatility filter and volume confirmation
# Donchian breakouts capture momentum; 1d ATR filter avoids high volatility choppy markets
# Volume spike confirms institutional participation
# Works in both bull and bear markets by trading breakouts in direction of higher timeframe trend
# Target: 20-50 trades/year (80-200 over 4 years) to minimize fee drag

name = "4h_Donchian20_1dATR_Filter_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for ATR filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # Align with original array
    
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Donchian channels (20-period)
    upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.8 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(50, 20)  # Need sufficient history for ATR and Donchian
    
    for i in range(start_idx, n):
        if (np.isnan(atr_14_1d_aligned[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when ATR is below 80th percentile (avoid choppy markets)
        # Calculate rolling percentile of ATR
        if i >= 50:
            atr_slice = atr_14_1d_aligned[max(0, i-49):i+1]
            atr_slice = atr_slice[~np.isnan(atr_slice)]
            if len(atr_slice) >= 10:
                atr_percentile = (np.sum(atr_slice <= atr_14_1d_aligned[i]) / len(atr_slice)) * 100
                low_volatility = atr_percentile < 80  # Only trade in lower 80% of volatility
            else:
                low_volatility = True  # Not enough data, allow trade
        else:
            low_volatility = True
        
        # Donchian breakout conditions
        breakout_up = close[i] > upper[i-1]  # Close above previous upper band
        breakout_down = close[i] < lower[i-1]  # Close below previous lower band
        
        if position == 0:  # Flat - look for new entries
            # Long: Donchian breakout up, volume spike, low volatility
            if breakout_up and volume_spike[i] and low_volatility:
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakout down, volume spike, low volatility
            elif breakout_down and volume_spike[i] and low_volatility:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on Donchian breakdown
            if close[i] < lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on Donchian breakout
            if close[i] > upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals