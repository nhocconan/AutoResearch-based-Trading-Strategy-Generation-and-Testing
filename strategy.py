#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dATR_Trend_VolumeSpike_v2
Hypothesis: Camarilla R1/S1 breakout with 1d ATR-based trend filter, volume spike confirmation, and ATR stoploss.
Designed for low trade frequency (20-40/year) to avoid fee drag while capturing strong breakouts in trending markets.
Uses discrete position sizing (0.25) to minimize fee churn. Focus on BTC/ETH with SOL as secondary.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels and ATR-based trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # 1d ATR(14) for trend filter (strong trend when ATR rising)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 1d ATR trend: rising ATR indicates strong trend (current ATR > ATR 5 periods ago)
    atr_5ago = np.roll(atr_1d_aligned, 5)
    atr_5ago[:5] = np.nan
    atr_rising = atr_1d_aligned > atr_5ago
    
    # Calculate Camarilla levels from previous 1d bar
    high_1d_prev = np.roll(high_1d, 1)
    low_1d_prev = np.roll(low_1d, 1)
    close_1d_prev = np.roll(close_1d, 1)
    high_1d_prev[0] = np.nan
    low_1d_prev[0] = np.nan
    close_1d_prev[0] = np.nan
    
    # Camarilla R1 and S1 levels (tighter breakout levels for fewer trades)
    R1 = close_1d_prev + (high_1d_prev - low_1d_prev) * 1.1 / 12
    S1 = close_1d_prev - (high_1d_prev - low_1d_prev) * 1.1 / 12
    
    # Align Camarilla levels
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Volume confirmation: 2.5x average volume (stricter to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # ATR for stoploss (using 14-period ATR)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of volume MA (24), ATR (14)
    start_idx = max(24, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(atr_rising[i]) or 
            np.isnan(R1_aligned[i]) or 
            np.isnan(S1_aligned[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(atr[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        atr_rising_val = atr_rising[i]
        R1_val = R1_aligned[i]
        S1_val = S1_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        volume_val = volume[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume confirmation and rising ATR (strong uptrend)
            long_signal = (high_val > R1_val) and (volume_val > 2.5 * vol_ma_val) and atr_rising_val
            # Short: price breaks below S1 with volume confirmation and rising ATR (strong downtrend)
            short_signal = (low_val < S1_val) and (volume_val > 2.5 * vol_ma_val) and atr_rising_val
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: ATR stoploss or trend weakening (ATR falling)
            if (close_val < entry_price - 2.5 * atr_val or 
                not atr_rising_val):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: ATR stoploss or trend weakening (ATR falling)
            if (close_val > entry_price + 2.5 * atr_val or 
                not atr_rising_val):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dATR_Trend_VolumeSpike_v2"
timeframe = "4h"
leverage = 1.0