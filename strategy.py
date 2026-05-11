#!/usr/bin/env python3
name = "6h_WeeklyPivot_BullBear_Switch_Trend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d High/Low/Close for weekly pivot
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Weekly pivot levels (using prior week's OHLC)
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L
    # S1 = 2*P - H
    # R2 = P + (H - L)
    # S2 = P - (H - L)
    # R3 = H + 2*(P - L)
    # S3 = L - 2*(H - P)
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    r1_1d = 2 * pivot_1d - low_1d
    s1_1d = 2 * pivot_1d - high_1d
    r2_1d = pivot_1d + range_1d
    s2_1d = pivot_1d - range_1d
    r3_1d = high_1d + 2 * (pivot_1d - low_1d)
    s3_1d = low_1d - 2 * (high_1d - pivot_1d)
    
    # Align 1d weekly pivot to 6h
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r2_1d_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    s2_1d_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # 1w trend filter: EMA50 on weekly close
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume filter: 6h volume > 1.5x 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > 1.5 * volume_ma20
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(pivot_1d_aligned[i]) or np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or
            np.isnan(r2_1d_aligned[i]) or np.isnan(s2_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or
            np.isnan(s3_1d_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Bull market: price above weekly EMA50
            # Go long on break above R2 with volume
            # Go short on break below S2 with volume
            if close[i] > ema_50_1w_aligned[i]:  # Bull regime
                if close[i] > r2_1d_aligned[i] and volume_filter[i]:
                    signals[i] = 0.25
                    position = 1
                elif close[i] < s2_1d_aligned[i] and volume_filter[i]:
                    signals[i] = -0.25
                    position = -1
            else:  # Bear regime
                # In bear market, fade at R3/S3 (stronger rejection)
                if close[i] > r3_1d_aligned[i] and volume_filter[i]:
                    signals[i] = -0.25  # Short at resistance
                    position = -1
                elif close[i] < s3_1d_aligned[i] and volume_filter[i]:
                    signals[i] = 0.25   # Long at support
                    position = 1
        elif position == 1:
            # Exit long: price breaks below R1 in bull, or S1 in bear
            if close[i] > ema_50_1w_aligned[i]:  # Still bull
                if close[i] < r1_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # Bear regime
                if close[i] > s1_1d_aligned[i]:  # Price above support, exit long
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above S1 in bull, or R1 in bear
            if close[i] > ema_50_1w_aligned[i]:  # Still bull
                if close[i] > s1_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:  # Bear regime
                if close[i] < r1_1d_aligned[i]:  # Price below resistance, exit short
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals