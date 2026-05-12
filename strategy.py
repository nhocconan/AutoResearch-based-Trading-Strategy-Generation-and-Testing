#!/usr/bin/env python3
"""
4h_Williams_Fractal_Breakout_1dTrend_VolumeFilter
Hypothesis: Price breaking above/below confirmed Williams fractals (high/low) on the 1d timeframe, with 1d EMA trend filter and volume confirmation (1.5x average) captures strong trend continuation while avoiding false breakouts. Fractals require 2-bar confirmation after the center bar, ensuring validity. Works in bull/bear by following 1d trend direction.
"""

name = "4h_Williams_Fractal_Breakout_1dTrend_VolumeFilter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')

    # Calculate Williams fractals on 1d data
    # Bearish fractal: high[n] > high[n-1] and high[n] > high[n+1]
    # Bullish fractal: low[n] < low[n-1] and low[n] < low[n+1]
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values

    bearish_fractal = np.full_like(high_1d, np.nan)
    bullish_fractal = np.full_like(low_1d, np.nan)

    for i in range(1, len(high_1d) - 1):
        if high_1d[i] > high_1d[i-1] and high_1d[i] > high_1d[i+1]:
            bearish_fractal[i] = high_1d[i]
        if low_1d[i] < low_1d[i-1] and low_1d[i] < low_1d[i+1]:
            bullish_fractal[i] = low_1d[i]

    # Require 2-bar confirmation after center bar for fractal validity
    bearish_fractal_confirmed = np.full_like(bearish_fractal, np.nan)
    bullish_fractal_confirmed = np.full_like(bullish_fractal, np.nan)

    for i in range(len(bearish_fractal)):
        if not np.isnan(bearish_fractal[i]):
            # Confirm 2 bars after the fractal bar
            if i + 2 < len(bearish_fractal):
                bearish_fractal_confirmed[i + 2] = bearish_fractal[i]
        if not np.isnan(bullish_fractal[i]):
            # Confirm 2 bars after the fractal bar
            if i + 2 < len(bullish_fractal):
                bullish_fractal_confirmed[i + 2] = bullish_fractal[i]

    # Align confirmed fractals to 4h timeframe with additional 2-bar delay
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal_confirmed, additional_delay_bars=0
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal_confirmed, additional_delay_bars=0
    )

    # 1d EMA34 trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)

    # Volume spike: >1.5x 20-period average (4h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(34, n):  # Start after EMA34 warmup
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above bullish fractal (support) + 1d EMA34 uptrend + volume spike
            if (close[i] > bullish_fractal_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below bearish fractal (resistance) + 1d EMA34 downtrend + volume spike
            elif (close[i] < bearish_fractal_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below bearish fractal (resistance break)
            if close[i] < bearish_fractal_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above bullish fractal (support break)
            if close[i] > bullish_fractal_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals