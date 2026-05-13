#!/usr/bin/env python3
# 1d_Weekly_Donchian_Breakout_RangeFilter_Volume
# Hypothesis: Use weekly Donchian channel (20-week) breakouts with 1d EMA200 trend filter and volume confirmation on daily timeframe.
# Long when price breaks above weekly upper band in uptrend with volume spike, short when price breaks below weekly lower band in downtrend with volume spike.
# Exit when price returns to weekly middle band or trend changes.
# Weekly Donchian provides robust breakout levels; EMA200 filters trend; volume confirms momentum.
# Designed for low trade frequency (30-100 total trades over 4 years) to minimize fee drag in ranging/bear markets.

name = "1d_Weekly_Donchian_Breakout_RangeFilter_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get weekly data for Donchian channel calculation
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly Donchian channel (20-period)
    high_20 = pd.Series(df_1w['high']).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(df_1w['low']).rolling(window=20, min_periods=20).min().values
    # Middle band = average of upper and lower
    mid_20 = (high_20 + low_20) / 2
    
    # Align weekly Donchian levels to daily timeframe
    upper_20_w = align_htf_to_ltf(prices, df_1w, high_20)
    lower_20_w = align_htf_to_ltf(prices, df_1w, low_20)
    mid_20_w = align_htf_to_ltf(prices, df_1w, mid_20)

    # Get daily data for EMA trend filter and volume average
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA200 for trend filter
    ema_200_1d = pd.Series(df_1d['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Volume filter: >1.8x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(upper_20_w[i]) or np.isnan(lower_20_w[i]) or 
            np.isnan(mid_20_w[i]) or np.isnan(ema_200_1d[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above weekly upper band + price above 1d EMA200 (uptrend) + volume spike
            if (close[i] > upper_20_w[i] and 
                close[i] > ema_200_1d[i] and
                volume[i] > vol_avg_20[i] * 1.8):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below weekly lower band + price below 1d EMA200 (downtrend) + volume spike
            elif (close[i] < lower_20_w[i] and 
                  close[i] < ema_200_1d[i] and
                  volume[i] > vol_avg_20[i] * 1.8):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to weekly middle band or trend changes (price below EMA200)
            if (close[i] <= mid_20_w[i] or close[i] < ema_200_1d[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to weekly middle band or trend changes (price above EMA200)
            if (close[i] >= mid_20_w[i] or close[i] > ema_200_1d[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals