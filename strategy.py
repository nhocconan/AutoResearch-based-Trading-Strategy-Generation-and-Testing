#!/usr/bin/env python3

# 6h_Pivot_Turn_Volume_Trend
# Hypothesis: On 6h timeframe, enter long when price retraces to prior 12h pivot low with volume >1.5x average and 12h EMA50 trending up; enter short when price retraces to prior 12h pivot high with volume >1.5x average and 12h EMA50 trending down.
# Uses 12h pivot points (high/low of prior bar) as dynamic support/resistance. The strategy fades overextensions in trending markets, which works in both bull and bear regimes. Targets 15-25 trades/year to minimize fee drag.

name = "6h_Pivot_Turn_Volume_Trend"
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

    # Get 12h data for pivot points and trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values

    # Calculate pivot points: use prior 12h bar's high and low
    pivot_high = np.roll(high_12h, 1)  # prior 12h high
    pivot_low = np.roll(low_12h, 1)    # prior 12h low
    pivot_high[0] = np.nan
    pivot_low[0] = np.nan

    # Align pivot points to 6h timeframe
    pivot_high_aligned = align_htf_to_ltf(prices, df_12h, pivot_high)
    pivot_low_aligned = align_htf_to_ltf(prices, df_12h, pivot_low)

    # 12h EMA50 for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)

    # Volume confirmation: volume > 1.5x 24-period average (approx 12 hours)
    vol_avg_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(pivot_high_aligned[i]) or np.isnan(pivot_low_aligned[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_avg_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price retraces to prior 12h pivot low + 12h uptrend + volume confirmation
            if (low[i] <= pivot_low_aligned[i] and 
                close[i] > pivot_low_aligned[i] and  # confirmation close above pivot low
                close[i] > ema50_12h_aligned[i] and
                volume[i] > vol_avg_24[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Price retraces to prior 12h pivot high + 12h downtrend + volume confirmation
            elif (high[i] >= pivot_high_aligned[i] and 
                  close[i] < pivot_high_aligned[i] and  # confirmation close below pivot high
                  close[i] < ema50_12h_aligned[i] and
                  volume[i] > vol_avg_24[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below prior 12h pivot low OR trend turns down
            if low[i] < pivot_low_aligned[i] or close[i] < ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above prior 12h pivot high OR trend turns up
            if high[i] > pivot_high_aligned[i] or close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals