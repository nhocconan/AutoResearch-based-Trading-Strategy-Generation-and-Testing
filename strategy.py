# 6h_WeeklyPivot_DonchianBreakout_TrendFilter
# Hypothesis: Use weekly pivot points to establish directional bias and 4-period Donchian breakout on 6h for entry.
# Weekly pivot provides strong support/resistance that works in both bull and bear markets.
# Breakout from recent 4-period high/low with momentum confirmation captures trend continuation.
# Designed for low trade frequency (12-37/year) to minimize fee drag on 6h timeframe.

name = "6h_WeeklyPivot_DonchianBreakout_TrendFilter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for pivot points
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 2:
        return np.zeros(n)
    
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Previous week's values for pivot calculation
    wh = np.concatenate([[weekly_high[0]], weekly_high[:-1]])
    wl = np.concatenate([[weekly_low[0]], weekly_low[:-1]])
    wc = np.concatenate([[weekly_close[0]], weekly_close[:-1]])
    
    # Calculate weekly pivot points
    pivot = (wh + wl + wc) / 3.0
    r1 = 2 * pivot - wl
    s1 = 2 * pivot - wh
    r2 = pivot + (wh - wl)
    s2 = pivot - (wh - wl)
    
    # Align weekly pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_weekly, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, r1)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, s1)
    r2_aligned = align_htf_to_ltf(prices, df_weekly, r2)
    s2_aligned = align_htf_to_ltf(prices, df_weekly, s2)
    
    # 4-period Donchian channels on 6h
    high_4 = np.full_like(high, np.nan)
    low_4 = np.full_like(low, np.nan)
    
    for i in range(4, len(high)):
        high_4[i] = np.max(high[i-3:i+1])
        low_4[i] = np.min(low[i-3:i+1])
    
    # Momentum confirmation: price change over 2 periods
    price_change = np.full_like(close, np.nan)
    for i in range(2, len(close)):
        price_change[i] = (close[i] - close[i-2]) / close[i-2]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(4, 2)  # Ensure Donchian and momentum are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(high_4[i]) or 
            np.isnan(low_4[i]) or np.isnan(price_change[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above 4-period high AND above weekly pivot AND positive momentum
            if (close[i] > high_4[i] and 
                close[i] > pivot_aligned[i] and 
                price_change[i] > 0):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below 4-period low AND below weekly pivot AND negative momentum
            elif (close[i] < low_4[i] and 
                  close[i] < pivot_aligned[i] and 
                  price_change[i] < 0):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below weekly support OR momentum turns negative
            if close[i] < s1_aligned[i] or price_change[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above weekly resistance OR momentum turns positive
            if close[i] > r1_aligned[i] or price_change[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals