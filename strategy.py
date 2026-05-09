#!/usr/bin/env python3
# Hypothesis: 12h timeframe with weekly pivot structure and weekly trend filter.
# Uses weekly Camarilla levels (R1/S1) for breakout entries and weekly EMA50 for trend filter.
# Weekly timeframe ensures alignment and reduces noise, providing robust structural levels.
# Target: 50-150 total trades over 4 years (12-37/year) with size 0.25.

name = "12h_Camarilla_R1_S1_1wEMA50_Trend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate weekly data for Camarilla levels and EMA
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Previous week's OHLC for Camarilla calculation
    prev_week_close = np.roll(df_1w['close'].values, 1)
    prev_week_high = np.roll(df_1w['high'].values, 1)
    prev_week_low = np.roll(df_1w['low'].values, 1)
    prev_week_close[0] = np.nan
    
    # Weekly Camarilla levels (R1, S1)
    camarilla_range = prev_week_high - prev_week_low
    r1 = prev_week_close + 1.1 * camarilla_range / 4
    s1 = prev_week_close - 1.1 * camarilla_range / 4
    
    # Align weekly levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # Weekly EMA50 trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume filter: current volume > 1.5x 20-period average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: breakout above R1 + weekly uptrend + volume filter
            if close[i] > r1_aligned[i] and close[i] > ema_50_1w_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: breakout below S1 + weekly downtrend + volume filter
            elif close[i] < s1_aligned[i] and close[i] < ema_50_1w_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to weekly EMA or breaks below S1
            if close[i] < ema_50_1w_aligned[i] or close[i] < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to weekly EMA or breaks above R1
            if close[i] > ema_50_1w_aligned[i] or close[i] > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals