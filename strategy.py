#!/usr/bin/env python3
name = "12h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "12h"
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
    
    # Load daily data for trend filter and pivot calculation
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Daily pivot points (using previous day's data)
    shift_high_1d = np.roll(high_1d, 1)
    shift_low_1d = np.roll(low_1d, 1)
    shift_close_1d = np.roll(close_1d, 1)
    shift_high_1d[0] = high_1d[0]
    shift_low_1d[0] = low_1d[0]
    shift_close_1d[0] = close_1d[0]
    
    daily_pivot = (shift_high_1d + shift_low_1d + shift_close_1d) / 3
    daily_range = shift_high_1d - shift_low_1d
    daily_r1 = daily_pivot + daily_range * 1.1 / 12
    daily_s1 = daily_pivot - daily_range * 1.1 / 12
    daily_r2 = daily_pivot + daily_range * 1.1 / 6
    daily_s2 = daily_pivot - daily_range * 1.1 / 6
    
    # Align daily pivot levels to 12h timeframe
    daily_r1_aligned = align_htf_to_ltf(prices, df_1d, daily_r1)
    daily_s1_aligned = align_htf_to_ltf(prices, df_1d, daily_s1)
    daily_r2_aligned = align_htf_to_ltf(prices, df_1d, daily_r2)
    daily_s2_aligned = align_htf_to_ltf(prices, df_1d, daily_s2)
    
    # Volume filter: current volume > 1.5x 24-period average (2 days of 12h data)
    vol_avg = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_filter = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(daily_r1_aligned[i]) or np.isnan(daily_s1_aligned[i]) or 
            np.isnan(daily_r2_aligned[i]) or np.isnan(daily_s2_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price rejects daily S1 (bounces off) + above daily EMA34 + volume filter
            # Rejection condition: low touches or goes below S1 but close recovers above S1
            if low[i] <= daily_s1_aligned[i] and close[i] > daily_s1_aligned[i] and close[i] > ema_34_1d_aligned[i] and vol_filter[i]:
                signals[i] = 0.30
                position = 1
            # Short: price rejects daily R1 (gets rejected) + below daily EMA34 + volume filter
            # Rejection condition: high touches or goes above R1 but close falls back below R1
            elif high[i] >= daily_r1_aligned[i] and close[i] < daily_r1_aligned[i] and close[i] < ema_34_1d_aligned[i] and vol_filter[i]:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price breaks below daily S2 or below daily EMA34
            if low[i] < daily_s2_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price breaks above daily R2 or above daily EMA34
            if high[i] > daily_r2_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals