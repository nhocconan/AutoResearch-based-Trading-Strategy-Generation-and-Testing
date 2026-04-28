#!/usr/bin/env python3
"""
6h_VolumeWeightedPrice_Action_WeeklyTrend
Hypothesis: On 6-hour timeframe, enter long when price closes above the 4-period volume-weighted high with bullish weekly trend (EMA13>EMA34), short when price closes below the 4-period volume-weighted low with bearish weekly trend (EMA13<EMA34). Exit on opposite signal. Uses volume weighting to filter weak moves and weekly trend to avoid counter-trend trades. Designed for moderate trade frequency (~20-40/year) to balance opportunity and cost in both bull and bear markets.
"""

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
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 34:
        return np.zeros(n)
    
    # Calculate weekly 13 and 34 EMA for trend filter
    close_weekly = df_weekly['close'].values
    ema13_weekly = pd.Series(close_weekly).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema34_weekly = pd.Series(close_weekly).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align weekly EMAs to 6h timeframe
    ema13_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema13_weekly)
    ema34_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema34_weekly)
    
    # Weekly trend: bullish when EMA13 > EMA34
    weekly_uptrend = ema13_weekly_aligned > ema34_weekly_aligned
    weekly_downtrend = ema13_weekly_aligned < ema34_weekly_aligned
    
    # Calculate 4-period volume-weighted high and low (using previous 4 periods, not including current)
    # Volume-weighted high: sum(high * volume) / sum(volume)
    vw_high = pd.Series(high * volume).rolling(window=4, min_periods=4).sum() / pd.Series(volume).rolling(window=4, min_periods=4).sum()
    vw_high = vw_high.shift(1).values  # Use previous period's value
    
    # Volume-weighted low: sum(low * volume) / sum(volume)
    vw_low = pd.Series(low * volume).rolling(window=4, min_periods=4).sum() / pd.Series(volume).rolling(window=4, min_periods=4).sum()
    vw_low = vw_low.shift(1).values  # Use previous period's value
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema13_weekly_aligned[i]) or np.isnan(ema34_weekly_aligned[i]) or
            np.isnan(vw_high[i]) or np.isnan(vw_low[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions with weekly trend alignment
        long_entry = close[i] > vw_high[i] and weekly_uptrend[i]
        short_entry = close[i] < vw_low[i] and weekly_downtrend[i]
        
        # Exit on opposite signal
        long_exit = close[i] < vw_low[i]
        short_exit = close[i] > vw_high[i]
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.25   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_VolumeWeightedPrice_Action_WeeklyTrend"
timeframe = "6h"
leverage = 1.0