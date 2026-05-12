#!/usr/bin/env python3
# 6h_Donchian20_WeeklyPivotDir_Volume
# Hypothesis: Use 20-period Donchian channel on 6h for breakout entries, filtered by weekly pivot direction (from 1d data aggregated to weekly) and volume confirmation (>2x 20-period average). Long when price breaks above upper band with bullish weekly pivot, short when breaks below lower band with bearish weekly pivot. Exit when price crosses the 6-period EMA. Designed for 6h timeframe to achieve 12-37 trades/year, working in both bull (breakouts) and bear (mean-reversion via pivots).

name = "6h_Donchian20_WeeklyPivotDir_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1d data to compute weekly pivot
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate weekly pivot from daily data (using last 5 days)
    # We'll compute pivot for each day based on prior 5-day window
    # PP = (H + L + C)/3 for the weekly pivot point
    # Using rolling window on daily data
    df_1d_df = pd.DataFrame({'high': high_1d, 'low': low_1d, 'close': close_1d})
    # Rolling window of 5 days for high, low, close
    rolling_high = df_1d_df['high'].rolling(window=5, min_periods=5).max().values
    rolling_low = df_1d_df['low'].rolling(window=5, min_periods=5).min().values
    rolling_close = df_1d_df['close'].rolling(window=5, min_periods=5).last().values  # last close in window
    # Weekly pivot point
    weekly_pp = (rolling_high + rolling_low + rolling_close) / 3.0
    # Weekly support/resistance (simplified: using 1-day range for weekly context)
    weekly_range = rolling_high - rolling_low
    weekly_r1 = close_1d + weekly_range * 1.1 / 12.0  # Adjusted for weekly context
    weekly_s1 = close_1d - weekly_range * 1.1 / 12.0

    # Align weekly pivot levels to 6h timeframe (available after weekly candle closes)
    # Note: We use the weekly pivot calculated from prior 5 days, so it's for the current week
    weekly_pp_aligned = align_htf_to_ltf(prices, df_1d, weekly_pp)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1d, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1d, weekly_s1)

    # 6h Donchian channel (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values

    # 6-period EMA for exit
    close_series = pd.Series(close)
    ema6 = close_series.ewm(span=6, adjust=False, min_periods=6).mean().values

    # Volume confirmation: volume > 2x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(weekly_pp_aligned[i]) or np.isnan(weekly_r1_aligned[i]) or
            np.isnan(weekly_s1_aligned[i]) or np.isnan(ema6[i]) or
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above Donchian high + price > weekly PP + volume spike
            if (close[i] > donchian_high[i] and
                close[i] > weekly_pp_aligned[i] and
                volume[i] > vol_avg_20[i] * 2.0):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian low + price < weekly PP + volume spike
            elif (close[i] < donchian_low[i] and
                  close[i] < weekly_pp_aligned[i] and
                  volume[i] > vol_avg_20[i] * 2.0):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below 6-period EMA
            if close[i] < ema6[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above 6-period EMA
            if close[i] > ema6[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals