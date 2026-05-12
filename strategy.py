#!/usr/bin/env python3
name = "6h_TurtleSoup_1dTrend_Filter"
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
    
    # Daily trend filter: EMA50
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Previous day's low and high for Turtle Soup setup
    prev_low_1d = np.roll(close_1d, 1)  # Approximate with previous close as proxy for simplicity
    prev_high_1d = np.roll(close_1d, 1)
    # Better: use actual low/high from daily data
    prev_low_1d = np.roll(df_1d['low'].values, 1)
    prev_high_1d = np.roll(df_1d['high'].values, 1)
    prev_low_1d_aligned = align_htf_to_ltf(prices, df_1d, prev_low_1d)
    prev_high_1d_aligned = align_htf_to_ltf(prices, df_1d, prev_high_1d)
    
    # 6-period lowest low and highest high for false breakout detection
    lowest_low_6 = pd.Series(low).rolling(window=6, min_periods=6).min().values
    highest_high_6 = pd.Series(high).rolling(window=6, min_periods=6).max().values
    
    # Volume filter: avoid low-volume breakouts
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(prev_low_1d_aligned[i]) or np.isnan(prev_high_1d_aligned[i]) or
            np.isnan(lowest_low_6[i]) or np.isnan(highest_high_6[i]) or
            np.isnan(vol_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long setup: false breakdown below previous day's low, then reversal
            # Price takes out prior day's low but closes back above it + 6-period lowest low
            if (low[i] < prev_low_1d_aligned[i] and 
                close[i] > prev_low_1d_aligned[i] and
                close[i] > lowest_low_6[i] and
                close[i] > ema_50_1d_aligned[i] and  # in line with daily trend
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short setup: false breakout above previous day's high, then reversal
            elif (high[i] > prev_high_1d_aligned[i] and 
                  close[i] < prev_high_1d_aligned[i] and
                  close[i] < highest_high_6[i] and
                  close[i] < ema_50_1d_aligned[i] and  # in line with daily trend
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below 6-period lowest low or reverses against daily trend
            if close[i] < lowest_low_6[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above 6-period highest high or reverses against daily trend
            if close[i] > highest_high_6[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals