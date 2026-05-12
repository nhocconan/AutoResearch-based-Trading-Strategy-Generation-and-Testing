#!/usr/bin/env python3
# 6h_1D_WilliamsFractal_Pullback_With_12h_Trend
# Hypothesis: Williams Fractals on 1d identify potential turning points; pullbacks in the direction of the 12h trend offer high-probability entries.
# Enter long when price pulls back to a bullish fractal low in a 12h uptrend; enter short when price pulls back to a bearish fractal high in a 12h downtrend.
# Uses 1d for fractal structure (key support/resistance) and 12h for trend filter (institutional bias).
# Works in bull markets (buy pullbacks in uptrends) and bear markets (sell rallies in downtrends) by following the higher-timeframe trend.

name = "6h_1D_WilliamsFractal_Pullback_With_12h_Trend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values

    # Get 1d data for Williams Fractals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)

    # Calculate Williams Fractals on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bearish_fractal, bullish_fractal = compute_williams_fractals(high_1d, low_1d)

    # Williams Fractals need 2 extra bars for confirmation (after the center bar)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)

    # Get 12h data for trend filter (EMA34)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 40:
        return np.zeros(n)

    close_12h = df_12h['close'].values
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(40, n):
        # Skip if any required data is NaN
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(ema34_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Pullback to bullish fractal low in 12h uptrend
            if bullish_fractal_aligned[i] > 0 and close[i] > ema34_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Pullback to bearish fractal high in 12h downtrend
            elif bearish_fractal_aligned[i] > 0 and close[i] < ema34_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below 12h EMA34 (trend change)
            if close[i] < ema34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above 12h EMA34 (trend change)
            if close[i] > ema34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals