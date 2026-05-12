# 1d_WeeklyPivot_Breakout_Trend_Volume
# Hypothesis: Weekly pivot levels act as strong support/resistance. Breakouts above R1/S1
# with weekly trend alignment and volume confirmation capture meaningful moves in both bull and bear markets.
# Timeframe: 1d balances trade frequency and signal quality, avoiding excessive churn.
# Weekly trend filter ensures we trade with the dominant weekly momentum, reducing whipsaws.
# Volume confirmation filters out low-conviction breakouts.
# Designed for 8-15 trades per year per symbol, well within the 30-100 target over 4 years.

#!/usr/bin/env python3
name = "1d_WeeklyPivot_Breakout_Trend_Volume"
timeframe = "1d"
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
    
    # Weekly data for pivot points and trend
    df_1w = get_htf_data(prices, '1w')
    high_w = df_1w['high'].values
    low_w = df_1w['low'].values
    close_w = df_1w['close'].values
    
    # Weekly pivot levels
    pivot_w = (high_w + low_w + close_w) / 3.0
    r1_w = 2 * pivot_w - low_w
    s1_w = 2 * pivot_w - high_w
    
    # Weekly trend: price above/below pivot
    weekly_trend_up = close_w > pivot_w
    weekly_trend_down = close_w < pivot_w
    
    # Align weekly data to daily timeframe
    pivot_w_aligned = align_htf_to_ltf(prices, df_1w, pivot_w)
    r1_w_aligned = align_htf_to_ltf(prices, df_1w, r1_w)
    s1_w_aligned = align_htf_to_ltf(prices, df_1w, s1_w)
    weekly_trend_up_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_up)
    weekly_trend_down_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_down)
    
    # Daily volume moving average for confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # ensure volume MA has enough data
    
    for i in range(start_idx, n):
        # Skip if weekly data not ready
        if np.isnan(pivot_w_aligned[i]) or np.isnan(r1_w_aligned[i]) or np.isnan(s1_w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Volume filter: current volume above 20-period average
        vol_filter = volume[i] > volume_ma[i] if not np.isnan(volume_ma[i]) else False
        
        if position == 0:
            # Long: weekly trend up + price breaks above R1 + volume confirmation
            if (weekly_trend_up_aligned[i] and 
                close[i] > r1_w_aligned[i] and 
                vol_filter):
                signals[i] = 0.25
                position = 1
            # Short: weekly trend down + price breaks below S1 + volume confirmation
            elif (weekly_trend_down_aligned[i] and 
                  close[i] < s1_w_aligned[i] and 
                  vol_filter):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below weekly pivot OR weekly trend changes to down
            if close[i] < pivot_w_aligned[i] or weekly_trend_down_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above weekly pivot OR weekly trend changes to up
            if close[i] > pivot_w_aligned[i] or weekly_trend_up_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals