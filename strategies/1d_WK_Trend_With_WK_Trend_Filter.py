#!/usr/bin/env python3
"""
1d_WK_Trend_With_WK_Trend_Filter
Hypothesis: On daily timeframe, use weekly trend (via 8/21 EMA crossover) as filter for entries triggered by price closing above/below prior day's high/low with volume confirmation. Weekly trend filter avoids counter-trend trades in choppy markets, while daily breakouts capture momentum. Volume surge confirms institutional participation. Designed for low trade frequency (<20/year) to minimize fee drag and work in both bull/bear markets via trend alignment.
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
    
    # Daily breakout: close above prior day's high or below prior day's low
    # Use shift(1) to access prior day's high/low (available at close of current day)
    prior_day_high = np.roll(high, 1)
    prior_day_low = np.roll(low, 1)
    # First bar has no prior day, set to current values to avoid false signals
    prior_day_high[0] = high[0]
    prior_day_low[0] = low[0]
    
    breakout_long = close > prior_day_high
    breakout_short = close < prior_day_low
    
    # Volume confirmation: current volume > 1.5x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_surge = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 21  # Wait for weekly EMA21 to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema8_weekly_aligned[i]) or np.isnan(ema21_weekly_aligned[i]) or
            np.isnan(volume_surge[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions with weekly trend alignment and volume surge
        long_entry = breakout_long[i] and weekly_uptrend[i] and volume_surge[i]
        short_entry = breakout_short[i] and weekly_downtrend[i] and volume_surge[i]
        
        # Exit on opposite breakout with volume surge (to avoid whipsaw)
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

name = "1d_WK_Trend_With_WK_Trend_Filter"
timeframe = "1d"
leverage = 1.0