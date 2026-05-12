#!/usr/bin/env python3
name = "6h_Donchian20_1dTrend_VolumeFilter_WeeklyPivot"
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
    
    # Load daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Load weekly data for pivot calculation
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
    weekly_r2 = weekly_pivot + weekly_range * 1.1 / 6
    weekly_s2 = weekly_pivot - weekly_range * 1.1 / 6
    
    # Align weekly levels to 6h timeframe
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    weekly_r2_aligned = align_htf_to_ltf(prices, df_1w, weekly_r2)
    weekly_s2_aligned = align_htf_to_ltf(prices, df_1w, weekly_s2)
    
    # Volume filter: current volume > 1.8x 24-period average (4 days of 6h data)
    vol_avg = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_filter = volume > (1.8 * vol_avg)
    
    # Donchian channel (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(weekly_r1_aligned[i]) or np.isnan(weekly_s1_aligned[i]) or 
            np.isnan(weekly_r2_aligned[i]) or np.isnan(weekly_s2_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high + above daily EMA50 + above weekly R1 + volume filter
            if (close[i] > donchian_high[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                close[i] > weekly_r1_aligned[i] and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low + below daily EMA50 + below weekly S1 + volume filter
            elif (close[i] < donchian_low[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  close[i] < weekly_s1_aligned[i] and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below Donchian low or below daily EMA50
            if close[i] < donchian_low[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above Donchian high or above daily EMA50
            if close[i] > donchian_high[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals