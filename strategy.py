#!/usr/bin/env python3
name = "1d_WeeklyPivot_Trend_Reversal_v1"
timeframe = "1d"
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
    
    # Load weekly data for pivot points and trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Weekly EMA34 for trend filter (using weekly data)
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Weekly pivot points (using previous week's data)
    shift_high_1w = np.roll(high_1w, 1)
    shift_low_1w = np.roll(low_1w, 1)
    shift_close_1w = np.roll(close_1w, 1)
    shift_high_1w[0] = high_1w[0]
    shift_low_1w[0] = low_1w[0]
    shift_close_1w[0] = close_1w[0]
    
    weekly_pivot = (shift_high_1w + shift_low_1w + shift_close_1w) / 3
    weekly_range = shift_high_1w - shift_low_1w
    weekly_r1 = weekly_pivot + weekly_range * 1.1 / 12
    weekly_s1 = weekly_pivot - weekly_range * 1.1 / 12
    weekly_r2 = weekly_pivot + weekly_range * 1.1 / 6
    weekly_s2 = weekly_pivot - weekly_range * 1.1 / 6
    
    # Align weekly pivot levels to daily timeframe
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    weekly_r2_aligned = align_htf_to_ltf(prices, df_1w, weekly_r2)
    weekly_s2_aligned = align_htf_to_ltf(prices, df_1w, weekly_s2)
    
    # Volume filter: current volume > 1.5x 10-period average (10 days)
    vol_avg = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    vol_filter = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(weekly_r1_aligned[i]) or np.isnan(weekly_s1_aligned[i]) or 
            np.isnan(weekly_r2_aligned[i]) or np.isnan(weekly_s2_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price rejects weekly S1 (bounces off) + above weekly EMA34 + volume filter
            if low[i] <= weekly_s1_aligned[i] and close[i] > weekly_s1_aligned[i] and close[i] > ema_34_1w_aligned[i] and vol_filter[i]:
                signals[i] = 0.30
                position = 1
            # Short: price rejects weekly R1 (gets rejected) + below weekly EMA34 + volume filter
            elif high[i] >= weekly_r1_aligned[i] and close[i] < weekly_r1_aligned[i] and close[i] < ema_34_1w_aligned[i] and vol_filter[i]:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price breaks below weekly S2 or below weekly EMA34
            if low[i] < weekly_s2_aligned[i] or close[i] < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price breaks above weekly R2 or above weekly EMA34
            if high[i] > weekly_r2_aligned[i] or close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals