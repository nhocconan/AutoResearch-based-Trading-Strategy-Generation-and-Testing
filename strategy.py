#!/usr/bin/env python3
"""
6h_1d_Camarilla_R3_S3_Breakout_1dTrend
Hypothesis: Uses Camarilla pivot levels from 1d (R3/S3) for breakout entries on 6h chart,
filtered by 1d EMA34 trend direction. Designed for low trade frequency by requiring
price to break key 1d pivot levels with trend alignment. Works in bull and bear
markets by following intermediate-term trend from 1d timeframe.
"""

name = "6h_1d_Camarilla_R3_S3_Breakout_1dTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 35:
        return np.zeros(n)
    
    # 6h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1d OHLCV for Camarilla Pivot Levels ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate pivot points using previous 1d's OHLC
    prev_high_1d = df_1d['high'].values
    prev_low_1d = df_1d['low'].values
    prev_close_1d = df_1d['close'].values
    
    pivot_1d = (prev_high_1d + prev_low_1d + prev_close_1d) / 3.0
    range_val_1d = prev_high_1d - prev_low_1d
    
    # Camarilla levels (R3 and S3)
    R3_1d = pivot_1d + (range_val_1d * 1.1 / 4)
    S3_1d = pivot_1d - (range_val_1d * 1.1 / 4)
    
    # Align to 6h timeframe
    R3_6h = align_htf_to_ltf(prices, df_1d, R3_1d)
    S3_6h = align_htf_to_ltf(prices, df_1d, S3_1d)
    
    # --- 1d EMA34 Trend Filter ---
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_6h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for EMA34 and pivot calculation)
    start_idx = 35
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN (first few bars)
        if (np.isnan(R3_6h[i]) or np.isnan(S3_6h[i]) or 
            np.isnan(ema_34_6h[i])):
            # Maintain position if valid, otherwise flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R3 with trend alignment
            if (close[i] > R3_6h[i] and 
                close[i] > ema_34_6h[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 with trend alignment
            elif (close[i] < S3_6h[i] and 
                  close[i] < ema_34_6h[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: opposite breakout
            if position == 1:
                # Exit long: price breaks below S3 (reversal signal)
                if close[i] < S3_6h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price breaks above R3 (reversal signal)
                if close[i] > R3_6h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals