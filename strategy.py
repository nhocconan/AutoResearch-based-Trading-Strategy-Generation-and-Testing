#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R + Weekly Pivot Trend Filter
Long when Williams %R < -80 (oversold) and weekly pivot trend is up (price > weekly pivot)
Short when Williams %R > -20 (overbought) and weekly pivot trend is down (price < weekly pivot)
Exit when Williams %R returns to neutral range (-50 to -50) or opposite extreme
Williams %R identifies mean-reversion opportunities, weekly pivot provides trend filter to avoid counter-trend trades
Designed for 6h timeframe with low trade frequency (target: 50-150 trades over 4 years)
"""

name = "6h_WilliamsR_WeeklyPivot_TrendFilter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate weekly pivot point (using previous week's OHLC)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Previous week's OHLC for pivot calculation
    prev_week_high = df_1w['high'].shift(1)
    prev_week_low = df_1w['low'].shift(1)
    prev_week_close = df_1w['close'].shift(1)
    
    # Calculate weekly pivot point
    weekly_pivot = (prev_week_high + prev_week_low + prev_week_close) / 3
    
    # Align weekly pivot to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot.values)
    
    # Calculate Williams %R (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    period = 14
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max()
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min()
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    williams_r = williams_r.values  # Convert to numpy array
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, period)  # Ensure enough data for Williams %R calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(weekly_pivot_aligned[i]) or np.isnan(williams_r[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Williams %R oversold (< -80) and price above weekly pivot (uptrend)
            if williams_r[i] < -80 and close[i] > weekly_pivot_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: Williams %R overbought (> -20) and price below weekly pivot (downtrend)
            elif williams_r[i] > -20 and close[i] < weekly_pivot_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R returns to neutral (> -50) or becomes overbought
            if williams_r[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R returns to neutral (< -50) or becomes oversold
            if williams_r[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals