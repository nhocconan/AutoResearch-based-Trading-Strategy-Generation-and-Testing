#!/usr/bin/env python3
# 6h_WilliamsFractal_Breakout_1dTrend_Volume
# Hypothesis: Enter long when price breaks above a confirmed bearish fractal (resistance) with 1d EMA uptrend and volume spike.
# Enter short when price breaks below a confirmed bullish fractal (support) with 1d EMA downtrend and volume spike.
# Exit when price returns to the midpoint between the last confirmed fractal levels.
# Uses daily Williams fractals with 2-bar confirmation to avoid look-ahead and focus on significant turning points.
# Targets 15-25 trades/year on 6h to minimize fee decay while capturing strong reversals in trending markets.
# Works in bull markets by catching breakouts and in bear markets by fading failed breaks at key levels.

name = "6h_WilliamsFractal_Breakout_1dTrend_Volume"
timeframe = "6h"
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

    # Get 1d data for Williams fractals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Williams Fractals: bearish (high) and bullish (low)
    # Bearish fractal: high[n-2] < high[n-1] > high[n] and high[n] > high[n+1] and high[n] > high[n+2]
    # Bullish fractal: low[n-2] > low[n-1] < low[n] and low[n] < low[n+1] and low[n] < low[n+2]
    n1d = len(high_1d)
    bearish_fractal = np.full(n1d, np.nan)
    bullish_fractal = np.full(n1d, np.nan)

    for i in range(2, n1d - 2):
        if (high_1d[i-2] < high_1d[i-1] and 
            high_1d[i] > high_1d[i-1] and 
            high_1d[i] > high_1d[i+1] and 
            high_1d[i] > high_1d[i+2]):
            bearish_fractal[i] = high_1d[i]
        if (low_1d[i-2] > low_1d[i-1] and 
            low_1d[i] < low_1d[i-1] and 
            low_1d[i] < low_1d[i+1] and 
            low_1d[i] < low_1d[i+2]):
            bullish_fractal[i] = low_1d[i]

    # Need 2 additional bars for confirmation (Williams fractal confirmation rule)
    bearish_fractal_confirmed = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_confirmed = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)

    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    # Volume confirmation: volume > 1.8x 30-period average
    vol_avg_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    last_bearish = np.nan  # Last confirmed bearish fractal level
    last_bullish = np.nan  # Last confirmed bullish fractal level

    for i in range(30, n):
        # Skip if any required value is NaN
        if (np.isnan(bearish_fractal_confirmed[i]) or np.isnan(bullish_fractal_confirmed[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_avg_30[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Update last confirmed fractal levels
        if not np.isnan(bearish_fractal_confirmed[i]):
            last_bearish = bearish_fractal_confirmed[i]
        if not np.isnan(bullish_fractal_confirmed[i]):
            last_bullish = bullish_fractal_confirmed[i]

        if position == 0:
            # LONG: Price breaks above last confirmed bearish fractal + uptrend + volume spike
            if (not np.isnan(last_bearish) and 
                close[i] > last_bearish and 
                close[i] > ema34_1d_aligned[i] and
                volume[i] > vol_avg_30[i] * 1.8):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below last confirmed bullish fractal + downtrend + volume spike
            elif (not np.isnan(last_bullish) and 
                  close[i] < last_bullish and 
                  close[i] < ema34_1d_aligned[i] and
                  volume[i] > vol_avg_30[i] * 1.8):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to midpoint between last fractal levels
            if not np.isnan(last_bearish) and not np.isnan(last_bullish):
                midpoint = (last_bearish + last_bullish) / 2.0
                if close[i] < midpoint:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to midpoint between last fractal levels
            if not np.isnan(last_bearish) and not np.isnan(last_bullish):
                midpoint = (last_bearish + last_bullish) / 2.0
                if close[i] > midpoint:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                signals[i] = -0.25

    return signals