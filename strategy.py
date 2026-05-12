#!/usr/bin/env python3
name = "12h_Camarilla_R1_S1_Breakout_1wTrend_Volume"
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
    
    # Load weekly data for trend and pivot calculation
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly pivot points (using previous week's data to avoid look-ahead)
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
    
    # Align weekly levels to 12h timeframe
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    # Weekly trend: price above/below weekly EMA 50
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume filter: current volume > 1.8x 24-period average (12 days of 12h data)
    vol_avg = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_filter = volume > (1.8 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(weekly_r1_aligned[i]) or np.isnan(weekly_s1_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above weekly R1 + above weekly EMA50 + volume filter
            if close[i] > weekly_r1_aligned[i] and close[i] > ema_50_1w_aligned[i] and vol_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly S1 + below weekly EMA50 + volume filter
            elif close[i] < weekly_s1_aligned[i] and close[i] < ema_50_1w_aligned[i] and vol_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below weekly S1 or below weekly EMA50
            if close[i] < weekly_s1_aligned[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above weekly R1 or above weekly EMA50
            if close[i] > weekly_r1_aligned[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals