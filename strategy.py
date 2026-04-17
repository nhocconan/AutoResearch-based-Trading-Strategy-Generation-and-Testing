#!/usr/bin/env python3
"""
4h_Pivot_R1_S1_Breakout_Volume_Filter_v1
Daily Camarilla pivot levels (R1, S1) + volume confirmation + ATR stop.
Long when price breaks above R1 with volume > 1.5x average.
Short when price breaks below S1 with volume > 1.5x average.
Exit when price returns to pivot (PP) or reverses with volume confirmation.
Uses 1d timeframe for pivot levels. Designed to capture breakouts with institutional volume.
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === Daily Pivot Levels (Camarilla) ===
    df_1d = get_htf_data(prices, '1d')
    # Previous day's OHLC for pivot calculation
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Pivot point (PP)
    pp = (prev_high + prev_low + prev_close) / 3.0
    # Camarilla levels
    r1 = pp + 1.1 * (prev_high - prev_low) / 12
    s1 = pp - 1.1 * (prev_high - prev_low) / 12
    
    # Align to 4h timeframe (wait for daily close)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # === Volume Filter ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = 1.5 * vol_ma  # 1.5x average volume
    
    # === ATR for stop management (optional) ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(pp_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(vol_threshold[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price breaks above R1 with volume confirmation
            if (close[i] > r1_aligned[i] and 
                volume[i] > vol_threshold[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below S1 with volume confirmation
            elif (close[i] < s1_aligned[i] and 
                  volume[i] > vol_threshold[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price returns to pivot PP OR reverses with volume
            if (close[i] < pp_aligned[i] or 
                (close[i] < close[i-1] and volume[i] > vol_threshold[i])):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to pivot PP OR reverses with volume
            if (close[i] > pp_aligned[i] or 
                (close[i] > close[i-1] and volume[i] > vol_threshold[i])):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Pivot_R1_S1_Breakout_Volume_Filter_v1"
timeframe = "4h"
leverage = 1.0