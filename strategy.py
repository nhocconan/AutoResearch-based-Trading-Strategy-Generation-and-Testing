#!/usr/bin/env python3
name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_Volume"
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
    
    # Load daily data for Camarilla pivot and EMA
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Camarilla pivot levels using previous day's data to avoid look-ahead
    shift_high_1d = np.roll(high_1d, 1)
    shift_low_1d = np.roll(low_1d, 1)
    shift_close_1d = np.roll(close_1d, 1)
    shift_high_1d[0] = high_1d[0]
    shift_low_1d[0] = low_1d[0]
    shift_close_1d[0] = close_1d[0]
    
    daily_pivot = (shift_high_1d + shift_low_1d + shift_close_1d) / 3
    daily_range = shift_high_1d - shift_low_1d
    r1 = daily_pivot + daily_range * 1.1 / 12
    s1 = daily_pivot - daily_range * 1.1 / 12
    
    # Align daily levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume filter: current volume > 1.5x 20-period average (5 days of 4h data)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 + above weekly EMA34 + volume filter
            if close[i] > r1_aligned[i] and close[i] > ema_34_1w_aligned[i] and vol_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + below weekly EMA34 + volume filter
            elif close[i] < s1_aligned[i] and close[i] < ema_34_1w_aligned[i] and vol_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below S1 or below weekly EMA34
            if close[i] < s1_aligned[i] or close[i] < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above R1 or above weekly EMA34
            if close[i] > r1_aligned[i] or close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals