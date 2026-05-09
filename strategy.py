#!/usr/bin/env python3
# Hypothesis: 1d timeframe with weekly pivot structure and weekly trend filter.
# Uses weekly Camarilla levels (R1/S1) for breakout entries and weekly EMA34 for trend filter.
# Weekly trend filter ensures alignment with higher timeframe momentum, reducing whipsaw.
# Target: 30-100 total trades over 4 years (7-25/year) with size 0.25.

name = "1d_Camarilla_R1_S1_1wEMA34_Trend"
timeframe = "1d"
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
    
    # Calculate weekly data for Camarilla levels and EMA34
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Weekly high, low, close for Camarilla calculation (using previous week's data)
    prev_week_high = np.roll(df_1w['high'].values, 1)
    prev_week_low = np.roll(df_1w['low'].values, 1)
    prev_week_close = np.roll(df_1w['close'].values, 1)
    
    # First value invalid due to roll
    prev_week_high[0] = np.nan
    prev_week_low[0] = np.nan
    prev_week_close[0] = np.nan
    
    # Weekly Camarilla levels: R1 and S1
    camarilla_range = prev_week_high - prev_week_low
    r1 = prev_week_close + 1.1 * camarilla_range / 4
    s1 = prev_week_close - 1.1 * camarilla_range / 4
    
    # Align weekly Camarilla levels to daily timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # Weekly EMA34 trend filter
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume filter: current volume > 1.5x 20-day average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need enough data for volume average
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: breakout above R1 + weekly uptrend + volume filter
            if close[i] > r1_aligned[i] and close[i] > ema_34_1w_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: breakout below S1 + weekly downtrend + volume filter
            elif close[i] < s1_aligned[i] and close[i] < ema_34_1w_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to weekly EMA34 or breakdown below S1
            if close[i] <= ema_34_1w_aligned[i] or close[i] < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to weekly EMA34 or breakout above R1
            if close[i] >= ema_34_1w_aligned[i] or close[i] > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals