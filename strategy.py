#!/usr/bin/env python3
# 12h_PriceChannel_Breakout_WeeklyTrend_Volume
# Hypothesis: Use weekly price channels (Donchian high/low) for breakout entries with daily trend filter and volume confirmation.
# Long when price breaks above weekly Donchian high in uptrend with volume spike, short when price breaks below weekly Donchian low in downtrend with volume spike.
# Exit when price returns to the weekly median (midpoint of channel) or trend changes.
# Weekly trend filter ensures alignment with higher timeframe momentum, reducing false breakouts in sideways markets.
# Volume confirmation filters out low-momentum breakouts.
# Designed for 12h timeframe with moderate trade frequency (50-150 total trades over 4 years).

name = "12h_PriceChannel_Breakout_WeeklyTrend_Volume"
timeframe = "12h"
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

    # Get weekly data for Donchian channel calculation
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly Donchian channel (20-period high/low)
    # Upper band = max(high, 20), Lower band = min(low, 20), Median = (upper + lower)/2
    high_series = pd.Series(df_weekly['high'])
    low_series = pd.Series(df_weekly['low'])
    weekly_high = high_series.rolling(window=20, min_periods=20).max().values
    weekly_low = low_series.rolling(window=20, min_periods=20).min().values
    weekly_median = (weekly_high + weekly_low) / 2.0
    
    # Align weekly Donchian levels to 12h timeframe
    weekly_high_aligned = align_htf_to_ltf(prices, df_weekly, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_weekly, weekly_low)
    weekly_median_aligned = align_htf_to_ltf(prices, df_weekly, weekly_median)

    # Get daily data for EMA trend filter
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate daily EMA50 for trend filter
    ema_50_daily = pd.Series(df_daily['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_daily_aligned = align_htf_to_ltf(prices, df_daily, ema_50_daily)

    # Volume filter: >1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(weekly_high_aligned[i]) or np.isnan(weekly_low_aligned[i]) or 
            np.isnan(weekly_median_aligned[i]) or np.isnan(ema_50_daily_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above weekly high + price above daily EMA50 (uptrend) + volume spike
            if (close[i] > weekly_high_aligned[i] and 
                close[i] > ema_50_daily_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below weekly low + price below daily EMA50 (downtrend) + volume spike
            elif (close[i] < weekly_low_aligned[i] and 
                  close[i] < ema_50_daily_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to weekly median or trend changes (price below EMA50)
            if (close[i] <= weekly_median_aligned[i] or close[i] < ema_50_daily_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to weekly median or trend changes (price above EMA50)
            if (close[i] >= weekly_median_aligned[i] or close[i] > ema_50_daily_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals