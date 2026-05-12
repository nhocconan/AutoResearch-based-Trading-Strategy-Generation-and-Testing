#!/usr/bin/env python3
name = "6h_WeeklyPivot_Breakout_DailyTrend_Volume"
timeframe = "6h"
leverage = 1.0

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
    
    # Load weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot (Monday's weekly candle)
    # Pivot = (High + Low + Close) / 3
    # R1 = Close + (High - Low) * 1.1 / 12
    # S1 = Close - (High - Low) * 1.1 / 12
    weekly_pivot = (high_1w + low_1w + close_1w) / 3
    weekly_r1 = close_1w + (high_1w - low_1w) * 1.1 / 12
    weekly_s1 = close_1w - (high_1w - low_1w) * 1.1 / 12
    
    # Align weekly pivot levels to 6h timeframe (use previous week's levels)
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    # Load daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume filter: current volume > 1.8x 24-period average (6h * 24 = 6 days)
    vol_avg = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_filter = volume > (1.8 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(pivot_1w_aligned[i]) or 
            np.isnan(r1_1w_aligned[i]) or
            np.isnan(s1_1w_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(vol_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above weekly R1 + above daily EMA34 + volume spike
            if (close[i] > r1_1w_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly S1 + below daily EMA34 + volume spike
            elif (close[i] < s1_1w_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below weekly S1 or below daily EMA34
            if close[i] < s1_1w_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above weekly R1 or above daily EMA34
            if close[i] > r1_1w_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals