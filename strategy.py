#!/usr/bin/env python3
name = "12h_1w_1d_Camarilla_R1S1_Breakout_Trend_Volume"
timeframe = "12h"
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
    
    # Get daily and weekly data
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 10 or len(df_1w) < 10:
        return np.zeros(n)
    
    # Daily Camarilla pivot levels (R1, S1)
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Pivot point and support/resistance levels
    pivot = (daily_high + daily_low + daily_close) / 3
    range_hl = daily_high - daily_low
    r1 = pivot + range_hl * 1.1 / 12
    s1 = pivot - range_hl * 1.1 / 12
    
    # Weekly EMA50 for trend filter
    weekly_close = df_1w['close'].values
    ema50_w = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_uptrend = weekly_close > ema50_w
    
    # Daily volume average for confirmation
    daily_volume = df_1d['volume'].values
    vol_ma20 = np.zeros(len(daily_volume))
    for i in range(len(daily_volume)):
        if i < 20:
            vol_ma20[i] = np.mean(daily_volume[:i+1]) if i > 0 else 0
        else:
            vol_ma20[i] = np.mean(daily_volume[i-19:i+1])
    
    # Align all to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend)
    vol_ma20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 20)
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or
            np.isnan(weekly_uptrend_aligned[i]) or
            np.isnan(vol_ma20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 + weekly uptrend + volume confirmation
            if (high[i] > r1_aligned[i] and 
                weekly_uptrend_aligned[i] and 
                volume[i] > 1.5 * vol_ma20_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + weekly downtrend + volume confirmation
            elif (low[i] < s1_aligned[i] and 
                  not weekly_uptrend_aligned[i] and 
                  volume[i] > 1.5 * vol_ma20_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below S1 or weekly trend turns down
            if (close[i] < s1_aligned[i] or not weekly_uptrend_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above R1 or weekly trend turns up
            if (close[i] > r1_aligned[i] or weekly_uptrend_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals