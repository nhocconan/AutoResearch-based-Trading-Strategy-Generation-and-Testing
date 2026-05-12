#!/usr/bin/env python3
# 12h_WilliamsFractal_Trend_Breakout
# Hypothesis: Use Williams fractal breakouts on 12h with 1w trend filter (EMA50) and volume confirmation. Enter long when price breaks above recent bearish fractal resistance in bullish 1w trend, short when price breaks below recent bullish fractal support in bearish 1w trend. Williams fractals require 2-bar confirmation, so we use additional_delay_bars=2. Targets 12-30 trades/year to minimize fee decay while capturing major trend turns.

name = "12h_WilliamsFractal_Trend_Breakout"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values

    # Get 1d data for Williams fractals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values

    # 1w EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)

    # Williams fractals on 1d (requires 2-bar confirmation after center)
    bearish_fractal, bullish_fractal = compute_williams_fractals(high_1d, low_1d)
    # Bearish fractal: high[center] > high[center-2] and high[center] > high[center+2]
    # Bullish fractal: low[center] < low[center-2] and low[center] < low[center+2]
    # Need 2 extra bars after center bar for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)

    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN
        if (np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: price > recent bearish fractal resistance (short-term resistance broken) +
            #       1w trend bullish (price > EMA50) + volume spike
            if (close[i] > bearish_fractal_aligned[i] and 
                close[i] > ema50_1w_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: price < recent bullish fractal support (short-term support broken) +
            #        1w trend bearish (price < EMA50) + volume spike
            elif (close[i] < bullish_fractal_aligned[i] and 
                  close[i] < ema50_1w_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price < recent bullish fractal support (trend weakening)
            if close[i] < bullish_fractal_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price > recent bearish fractal resistance (trend weakening)
            if close[i] > bearish_fractal_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals