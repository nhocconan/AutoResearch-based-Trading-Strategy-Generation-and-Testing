#!/usr/bin/env python3
name = "6h_WeeklyPivot_DailyTrend_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data once for pivot levels (weekly high/low/close)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot point and support/resistance (previous week)
    # Standard pivot: P = (H + L + C)/3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    pw = (high_1w + low_1w + close_1w) / 3
    r1w = 2 * pw - low_1w
    s1w = 2 * pw - high_1w
    r2w = pw + (high_1w - low_1w)
    s2w = pw - (high_1w - low_1w)
    
    # Align weekly pivots to 6h (wait for weekly close)
    r1w_aligned = align_htf_to_ltf(prices, df_1w, r1w)
    s1w_aligned = align_htf_to_ltf(prices, df_1w, s1w)
    r2w_aligned = align_htf_to_ltf(prices, df_1w, r2w)
    s2w_aligned = align_htf_to_ltf(prices, df_1w, s2w)
    
    # Load daily data for trend filter (EMA 34)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r1w_aligned[i]) or np.isnan(s1w_aligned[i]) or 
            np.isnan(r2w_aligned[i]) or np.isnan(s2w_aligned[i]) or
            np.isnan(ema_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R2 weekly pivot + above daily EMA34 + volume filter
            if (close[i] > r2w_aligned[i] and close[i] > ema_1d_aligned[i] and vol_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S2 weekly pivot + below daily EMA34 + volume filter
            elif (close[i] < s2w_aligned[i] and close[i] < ema_1d_aligned[i] and vol_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below S1 weekly pivot
            if close[i] < s1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above R1 weekly pivot
            if close[i] > r1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals