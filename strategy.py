#!/usr/bin/env python3
# 1d_1w_Donchian_Breakout_Trend_Filter
# Hypothesis: Uses 1-week Donchian channels to establish trend direction and 1-day Donchian breakouts for entries.
# The 1-week channel acts as a trend filter (price above/below weekly high/low), while 1-day breakouts provide entry timing.
# Volume confirmation (>1.5x 20-day average) ensures institutional participation.
# Designed for low trade frequency (<50 total trades over 4 years) to minimize fee drift.
# Works in bull markets (breakouts above weekly high) and bear markets (breakdowns below weekly low).

name = "1d_1w_Donchian_Breakout_Trend_Filter"
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
    
    # Volume spike: >1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Weekly data for trend filter (Donchian channels)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly Donchian channels (20-period)
    def donchian_channels(high_arr, low_arr, window):
        upper = pd.Series(high_arr).rolling(window=window, min_periods=window).max().values
        lower = pd.Series(low_arr).rolling(window=window, min_periods=window).min().values
        return upper, lower
    
    weekly_upper, weekly_lower = donchian_channels(high_1w, low_1w, 20)
    
    # Daily Donchian channels (20-period) for entry signals
    daily_upper, daily_lower = donchian_channels(high, low, 20)
    
    # Align weekly Donchian channels to daily timeframe
    weekly_upper_aligned = align_htf_to_ltf(prices, df_1w, weekly_upper)
    weekly_lower_aligned = align_htf_to_ltf(prices, df_1w, weekly_lower)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        if (np.isnan(weekly_upper_aligned[i]) or
            np.isnan(weekly_lower_aligned[i]) or
            np.isnan(daily_upper[i]) or
            np.isnan(daily_lower[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price above weekly upper AND breaks above daily upper + volume spike
            if (close[i] > weekly_upper_aligned[i] and
                close[i] > daily_upper[i] and
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price below weekly lower AND breaks below daily lower + volume spike
            elif (close[i] < weekly_lower_aligned[i] and
                  close[i] < daily_lower[i] and
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below daily lower OR weekly trend turns bearish
            if (close[i] < daily_lower[i]) or \
               (close[i] < weekly_lower_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above daily upper OR weekly trend turns bullish
            if (close[i] > daily_upper[i]) or \
               (close[i] > weekly_upper_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals