#!/usr/bin/env python3
"""
1d_Donchian20_WeeklyTrend
Hypothesis: On daily timeframe, buy when price breaks above weekly Donchian(20) high with volume >2x average and weekly close > weekly open (bullish weekly candle); sell when price breaks below weekly Donchian(20) low with volume >2x average and weekly close < weekly open (bearish weekly candle). Uses weekly trend filter and volume confirmation to capture strong trends while minimizing false breakouts. Targets 15-25 trades per year to reduce fee drift and survive bear markets.
"""

name = "1d_Donchian20_WeeklyTrend"
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

    # Get weekly data for Donchian channel and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    open_1w = df_1w['open'].values

    # Weekly Donchian(20) high/low (use previous 20 weekly bars)
    def donchian_channel(arr, window):
        dcn = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            dcn[i] = np.max(arr[i-window+1:i+1])
        dcl = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            dcl[i] = np.min(arr[i-window+1:i+1])
        return dcn, dcl

    donchian_high, donchian_low = donchian_channel(high_1w, 20)
    donchian_low = donchian_channel(low_1w, 20)[1]  # Reuse function for low

    # Weekly trend: bullish if close > open, bearish if close < open
    weekly_bullish = close_1w > open_1w
    weekly_bearish = close_1w < open_1w

    # Align weekly data to daily timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish.astype(float))

    # Volume confirmation: volume > 2x 20-day average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(weekly_bullish_aligned[i]) or np.isnan(weekly_bearish_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above weekly Donchian high + weekly bullish + volume spike
            if (close[i] > donchian_high_aligned[i] and 
                weekly_bullish_aligned[i] > 0.5 and 
                volume[i] > vol_avg_20[i] * 2.0):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below weekly Donchian low + weekly bearish + volume spike
            elif (close[i] < donchian_low_aligned[i] and 
                  weekly_bearish_aligned[i] > 0.5 and 
                  volume[i] > vol_avg_20[i] * 2.0):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below weekly Donchian low
            if close[i] < donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above weekly Donchian high
            if close[i] > donchian_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals