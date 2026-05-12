#!/usr/bin/env python3
name = "4h_Camarilla_R1_S1_Breakout_12hEMA34_Trend_Volume"
timeframe = "4h"
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
    
    # Load daily data for pivot points (previous day's data)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate daily pivot points (using previous day's data)
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
    
    # Align daily pivot levels to 4h timeframe
    daily_r1_aligned = align_htf_to_ltf(prices, df_1d, daily_r1)
    daily_s1_aligned = align_htf_to_ltf(prices, df_1d, daily_s1)
    daily_r2_aligned = align_htf_to_ltf(prices, df_1d, daily_r2)
    daily_s2_aligned = align_htf_to_ltf(prices, df_1d, daily_s2)
    
    # Load 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # 12h EMA34 for trend filter
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Volume filter: current volume > 1.5x 20-period average (approx 5 days of 4h data)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(daily_r1_aligned[i]) or np.isnan(daily_s1_aligned[i]) or 
            np.isnan(daily_r2_aligned[i]) or np.isnan(daily_s2_aligned[i]) or 
            np.isnan(ema_34_12h_aligned[i]) or np.isnan(vol_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price rejects daily S1 (bounces off) + above 12h EMA34 + volume filter
            # Rejection condition: low touches or goes below S1 but close recovers above S1
            if low[i] <= daily_s1_aligned[i] and close[i] > daily_s1_aligned[i] and close[i] > ema_34_12h_aligned[i] and vol_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price rejects daily R1 (gets rejected) + below 12h EMA34 + volume filter
            # Rejection condition: high touches or goes above R1 but close falls back below R1
            elif high[i] >= daily_r1_aligned[i] and close[i] < daily_r1_aligned[i] and close[i] < ema_34_12h_aligned[i] and vol_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below daily S2 or below 12h EMA34
            if low[i] < daily_s2_aligned[i] or close[i] < ema_34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above daily R2 or above 12h EMA34
            if high[i] > daily_r2_aligned[i] or close[i] > ema_34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals