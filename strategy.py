#!/usr/bin/env python3
"""
6h_Donchian20_WeeklyTrend_Breakout
Hypothesis: On 6-hour timeframe, use Donchian(20) breakouts in the direction of weekly trend (via 8/21 EMA crossover) with volume confirmation. Weekly trend filter avoids counter-trend trades during extended trends, while Donchian breakouts capture momentum bursts. Volume surge confirms institutional participation. Designed for moderate trade frequency (~30-60/year) to balance opportunity and fee decay in both bull and bear markets.
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
    
    # Align weekly EMAs to 6h timeframe
    ema8_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema8_weekly)
    ema21_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema21_weekly)
    
    # Weekly trend: bullish when EMA8 > EMA21
    weekly_uptrend = ema8_weekly_aligned > ema21_weekly_aligned
    weekly_downtrend = ema8_weekly_aligned < ema21_weekly_aligned
    
    # Donchian channels (20-period) - using prior bars only
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Donchian breakouts
    breakout_long = close > highest_high
    breakout_short = close < lowest_low
    
    # Volume confirmation: current volume > 1.8x 50-period average
    vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_surge = volume > (vol_ma_50 * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 70  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema8_weekly_aligned[i]) or np.isnan(ema21_weekly_aligned[i]) or
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(volume_surge[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions with weekly trend alignment and volume surge
        long_entry = breakout_long[i] and weekly_uptrend[i] and volume_surge[i]
        short_entry = breakout_short[i] and weekly_downtrend[i] and volume_surge[i]
        
        # Exit on opposite breakout with volume surge
        long_exit = breakout_short[i] and volume_surge[i]
        short_exit = breakout_long[i] and volume_surge[i]
        
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

name = "6h_Donchian20_WeeklyTrend_Breakout"
timeframe = "6h"
leverage = 1.0