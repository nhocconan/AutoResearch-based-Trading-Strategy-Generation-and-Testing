#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull/Bear Power) with weekly trend filter.
# Elder Ray measures bull/bear power relative to EMA13; combined with weekly EMA34 trend.
# Long when Bull Power > 0 and weekly EMA34 rising; Short when Bear Power < 0 and weekly EMA34 falling.
# Weekly trend filter reduces whipsaw in sideways markets, focusing on stronger trends.
# Designed for low frequency (~10-20 trades/year) with clear trend-following edge.
name = "6h_ElderRay_WeeklyTrend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate EMA13 for Elder Ray
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13  # Bull Power: High - EMA13
    bear_power = low - ema13   # Bear Power: Low - EMA13
    
    # Weekly trend filter: EMA34 on weekly timeframe
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    weekly_close = df_1w['close'].values
    weekly_ema34 = pd.Series(weekly_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    weekly_ema34_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema34)
    
    # Weekly trend: rising if current > previous, falling if current < previous
    weekly_trend_up = np.zeros(n, dtype=bool)
    weekly_trend_down = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if not np.isnan(weekly_ema34_aligned[i]) and not np.isnan(weekly_ema34_aligned[i-1]):
            weekly_trend_up[i] = weekly_ema34_aligned[i] > weekly_ema34_aligned[i-1]
            weekly_trend_down[i] = weekly_ema34_aligned[i] < weekly_ema34_aligned[i-1]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Ensure enough data for weekly EMA34
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(weekly_ema34_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Bull Power positive AND weekly trend up
            if bull_power[i] > 0 and weekly_trend_up[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power negative AND weekly trend down
            elif bear_power[i] < 0 and weekly_trend_down[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if Bull Power turns negative or weekly trend turns down
            if bull_power[i] <= 0 or not weekly_trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if Bear Power turns positive or weekly trend turns up
            if bear_power[i] >= 0 or not weekly_trend_down[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals