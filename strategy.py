#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h1d_Pivot_R1S1_Breakout_Volume_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h and 1d data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # 4h pivot levels from previous 4h bar
    prev_close_4h = df_4h['close'].shift(1).values
    prev_high_4h = df_4h['high'].shift(1).values
    prev_low_4h = df_4h['low'].shift(1).values
    pivot_4h = (prev_high_4h + prev_low_4h + prev_close_4h) / 3
    range_4h = prev_high_4h - prev_low_4h
    R1_4h = pivot_4h + range_4h * 0.1
    S1_4h = pivot_4h - range_4h * 0.1
    
    # 1d pivot levels from previous 1d bar
    prev_close_1d = df_1d['close'].shift(1).values
    prev_high_1d = df_1d['high'].shift(1).values
    prev_low_1d = df_1d['low'].shift(1).values
    pivot_1d = (prev_high_1d + prev_low_1d + prev_close_1d) / 3
    range_1d = prev_high_1d - prev_low_1d
    R1_1d = pivot_1d + range_1d * 0.1
    S1_1d = pivot_1d - range_1d * 0.1
    
    # Align HTF pivot levels to 1h (wait for HTF close)
    R1_4h_aligned = align_htf_to_ltf(prices, df_4h, R1_4h)
    S1_4h_aligned = align_htf_to_ltf(prices, df_4h, S1_4h)
    R1_1d_aligned = align_htf_to_ltf(prices, df_1d, R1_1d)
    S1_1d_aligned = align_htf_to_ltf(prices, df_1d, S1_1d)
    
    # Volume filter: current volume > 2.0 * 24-period average (1 day)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > (2.0 * vol_ma_24)
    
    # Session filter: 8-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(R1_4h_aligned[i]) or np.isnan(S1_4h_aligned[i]) or
            np.isnan(R1_1d_aligned[i]) or np.isnan(S1_1d_aligned[i]) or
            np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        # Check session filter
        if not session_filter[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        close_val = close[i]
        R1_4h_val = R1_4h_aligned[i]
        S1_4h_val = S1_4h_aligned[i]
        R1_1d_val = R1_1d_aligned[i]
        S1_1d_val = S1_1d_aligned[i]
        vol_filter = volume_filter[i]
        
        # Entry conditions: break of both 4h and 1d R1/S1 with volume
        if position == 0:
            # Long: break above both R1 levels
            if close_val > R1_4h_val and close_val > R1_1d_val and vol_filter:
                signals[i] = 0.20
                position = 1
            # Short: break below both S1 levels
            elif close_val < S1_4h_val and close_val < S1_1d_val and vol_filter:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: price falls back below either S1 level
            if close_val < S1_4h_val or close_val < S1_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: price rises back above either R1 level
            if close_val > R1_4h_val or close_val > R1_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals