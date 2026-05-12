#!/usr/bin/env python3
"""
1d_WeeklyTrend_PriceChannel_Retest
Hypothesis: Buy pullbacks in weekly uptrend to daily support/resistance zones; sell rallies in weekly downtrend to daily resistance/support zones. Uses weekly trend filter (price above/below weekly 50 EMA) and daily price channel (Donchian) retest with volume confirmation. Works in both bull and bear markets by only taking trades in the direction of the weekly trend, reducing counter-trend whipsaws. Target: 15-25 trades/year.
"""

name = "1d_WeeklyTrend_PriceChannel_Retest"
timeframe = "1d"
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

    # Get weekly data for trend filter (call once before loop)
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)
    close_weekly = df_weekly['close'].values
    ema50_weekly = pd.Series(close_weekly).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema50_weekly)

    # Daily Donchian channels (20-period)
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values

    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        ema50_val = ema50_weekly_aligned[i]
        hh20 = highest_high_20[i]
        ll20 = lowest_low_20[i]
        vol_avg_val = vol_avg_20[i]

        if np.isnan(ema50_val) or np.isnan(hh20) or np.isnan(ll20) or np.isnan(vol_avg_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Weekly uptrend (price > weekly EMA50) + price retests daily lower band (support) + volume
            if close[i] > ema50_val and close[i] <= ll20 * 1.005 and volume[i] > vol_avg_val * 1.5:
                signals[i] = 0.25
                position = 1
            # SHORT: Weekly downtrend (price < weekly EMA50) + price retests daily upper band (resistance) + volume
            elif close[i] < ema50_val and close[i] >= hh20 * 0.995 and volume[i] > vol_avg_val * 1.5:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reaches daily upper band (resistance) or weekly trend turns down
            if close[i] >= hh20 or close[i] < ema50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reaches daily lower band (support) or weekly trend turns up
            if close[i] <= ll20 or close[i] > ema50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals