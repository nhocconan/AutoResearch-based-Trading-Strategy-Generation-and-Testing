#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout (20) + 1d ATR filter + volume confirmation
# Donchian channels identify trend breakouts, 1d ATR filter ensures we trade only in sufficient volatility regimes,
# volume confirmation adds conviction. This combination has shown robustness across bull/bear markets.
# Target: 20-30 trades/year per symbol with disciplined entries
name = "4h_Donchian20_1dATR_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d ATR for volatility filter (use true range)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate True Range for daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Donchian channels (20-period) on 4h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(atr_1d_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Only trade when 1d ATR is above its 50-period average (sufficient volatility)
        if np.isnan(atr_1d_aligned[i]):
            signals[i] = 0.0
            continue
            
        atr_ma_50 = pd.Series(atr_1d_aligned).rolling(window=50, min_periods=50).mean().values[i]
        if np.isnan(atr_ma_50) or atr_1d_aligned[i] < atr_ma_50:
            # Low volatility regime - stay flat
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian upper band + volume confirmation
            if close[i] > highest_high[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower band + volume confirmation
            elif close[i] < lowest_low[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below Donchian lower band
            if close[i] < lowest_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above Donchian upper band
            if close[i] > highest_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals