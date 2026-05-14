#!/usr/bin/env python3
"""
1d_3BarBreakout_WeeklyTrend_200MA_Filter
Hypothesis: On daily timeframe, enter long when price breaks above the 3-day high with volume surge and weekly uptrend (EMA8>EMA21), short when price breaks below the 3-day low with volume surge and weekly downtrend. Exit on opposite breakout with volume. Price must be above/below 200-day MA to avoid counter-trend trades. Designed for low trade frequency (~10-25/year) to minimize fee decay in both bull and bear markets.
"""

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
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 21:
        return np.zeros(n)
    
    # Calculate weekly 8 and 21 EMA for trend filter
    close_weekly = df_weekly['close'].values
    ema8_weekly = pd.Series(close_weekly).ewm(span=8, adjust=False, min_periods=8).mean().values
    ema21_weekly = pd.Series(close_weekly).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align weekly EMAs to daily timeframe
    ema8_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema8_weekly)
    ema21_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema21_weekly)
    
    # Weekly trend: bullish when EMA8 > EMA21
    weekly_uptrend = ema8_weekly_aligned > ema21_weekly_aligned
    weekly_downtrend = ema8_weekly_aligned < ema21_weekly_aligned
    
    # Calculate 3-day high and low (using previous 3 days, not including current)
    high_3d = pd.Series(high).rolling(window=3, min_periods=3).max().shift(1).values
    low_3d = pd.Series(low).rolling(window=3, min_periods=3).min().shift(1).values
    
    # Volume confirmation: current volume > 2.0x 50-day average
    vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_surge = volume > (vol_ma_50 * 2.0)
    
    # 200-day moving average filter
    ma_200 = pd.Series(close).rolling(window=200, min_periods=200).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema8_weekly_aligned[i]) or np.isnan(ema21_weekly_aligned[i]) or
            np.isnan(high_3d[i]) or np.isnan(low_3d[i]) or np.isnan(volume_surge[i]) or
            np.isnan(ma_200[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions with weekly trend alignment, volume surge, and 200MA filter
        long_entry = close[i] > high_3d[i] and weekly_uptrend[i] and volume_surge[i] and close[i] > ma_200[i]
        short_entry = close[i] < low_3d[i] and weekly_downtrend[i] and volume_surge[i] and close[i] < ma_200[i]
        
        # Exit on opposite 3-day break with volume surge (no 200MA filter on exit to avoid whipsaw)
        long_exit = close[i] < low_3d[i] and volume_surge[i]
        short_exit = close[i] > high_3d[i] and volume_surge[i]
        
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

name = "1d_3BarBreakout_WeeklyTrend_200MA_Filter"
timeframe = "1d"
leverage = 1.0