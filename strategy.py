#!/usr/bin/env python3
# 1D_WeeklyDonchian_Breakout_Volume
# Hypothesis: Weekly Donchian channels identify long-term structural breakouts.
# A breakout above the weekly high (20-week high) or below the weekly low (20-week low)
# with volume confirmation signals a trend continuation.
# Works in both bull and bear markets: breakouts capture new trends, while volume
# filters avoid false signals in choppy conditions.
# Weekly timeframe reduces noise and false breakouts, leading to lower trade frequency.
# Position size: 0.25 for long/short, 0.0 for flat.

name = "1D_WeeklyDonchian_Breakout_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data (HTF)
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly Donchian channels (20-period)
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    
    # Weekly upper band: highest high over past 20 weeks
    weekly_high_max = pd.Series(weekly_high).rolling(window=20, min_periods=20).max().values
    # Weekly lower band: lowest low over past 20 weeks
    weekly_low_min = pd.Series(weekly_low).rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian levels to daily timeframe
    weekly_high_max_aligned = align_htf_to_ltf(prices, df_weekly, weekly_high_max)
    weekly_low_min_aligned = align_htf_to_ltf(prices, df_weekly, weekly_low_min)
    
    # Volume confirmation: 20-day average volume
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need at least 20 weeks of data for Donchian
    
    for i in range(start_idx, n):
        if np.isnan(weekly_high_max_aligned[i]) or np.isnan(weekly_low_min_aligned[i]) or \
           np.isnan(volume_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_confirm = volume[i] > 1.5 * volume_ma[i] if volume_ma[i] > 0 else False
        
        if position == 0:
            # Long breakout: price closes above weekly high AND volume confirmation
            if close[i] > weekly_high_max_aligned[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short breakout: price closes below weekly low AND volume confirmation
            elif close[i] < weekly_low_min_aligned[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below weekly low (breakdown of the trend)
            if close[i] < weekly_low_min_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above weekly high (breakout against the short)
            if close[i] > weekly_high_max_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals