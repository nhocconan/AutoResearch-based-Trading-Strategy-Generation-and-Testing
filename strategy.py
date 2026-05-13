#!/usr/bin/env python3
# 12h_Keltner_Channel_Breakout_1wTrend_Volume
# Hypothesis: Use weekly Keltner Channel breakouts with weekly EMA trend filter and volume confirmation.
# Enter long when price breaks above upper Keltner Channel (EMA20 + 2*ATR) with weekly EMA50 uptrend and volume spike.
# Enter short when price breaks below lower Keltner Channel (EMA20 - 2*ATR) with weekly EMA50 downtrend and volume spike.
# Exit when price crosses back below/above the EMA20 middle line.
# Weekly trend filter reduces false signals in ranging markets, volume confirmation ensures momentum.
# Designed for 12h timeframe to capture multi-day trends with low trade frequency (target: 15-25 trades/year).
# Works in both bull and bear markets by following the weekly trend direction.

name = "12h_Keltner_Channel_Breakout_1wTrend_Volume"
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

    # Get weekly data for Keltner Channel and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values

    # Calculate Keltner Channel: EMA20 ± 2*ATR(10)
    # EMA20
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    # ATR(10): True Range = max(high-low, |high-close_prev|, |low-close_prev|)
    tr1 = np.abs(high_1w - low_1w)
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = np.nan  # First value has no previous close
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr10_1w = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    # Upper and Lower Bands
    upper_1w = ema20_1w + 2 * atr10_1w
    lower_1w = ema20_1w - 2 * atr10_1w

    # Weekly EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values

    # Volume confirmation: weekly volume > 1.5x 20-period average
    vol_avg_20w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values

    # Align all weekly indicators to 12h timeframe
    upper_1w_aligned = align_htf_to_ltf(prices, df_1w, upper_1w)
    lower_1w_aligned = align_htf_to_ltf(prices, df_1w, lower_1w)
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    vol_avg_20w_aligned = align_htf_to_ltf(prices, df_1w, vol_avg_20w)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN
        if (np.isnan(upper_1w_aligned[i]) or np.isnan(lower_1w_aligned[i]) or 
            np.isnan(ema20_1w_aligned[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(vol_avg_20w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close breaks above upper Keltner + weekly EMA50 uptrend + volume spike
            if (close[i] > upper_1w_aligned[i] and 
                close[i] > ema50_1w_aligned[i] and
                volume[i] > vol_avg_20w_aligned[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Close breaks below lower Keltner + weekly EMA50 downtrend + volume spike
            elif (close[i] < lower_1w_aligned[i] and 
                  close[i] < ema50_1w_aligned[i] and
                  volume[i] > vol_avg_20w_aligned[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close crosses back below EMA20 (middle line)
            if close[i] < ema20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close crosses back above EMA20 (middle line)
            if close[i] > ema20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals