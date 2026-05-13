#!/usr/bin/env python3
# 4h_Donchian_Breakout_Volume_Regime
# Hypothesis: Use Donchian(20) breakout on 4h with volume confirmation and chop regime filter.
# Long when price breaks above upper band with volume > 1.5x 24-period avg and CHOP > 61.8 (range).
# Short when price breaks below lower band with volume > 1.5x 24-period avg and CHOP > 61.8.
# Exit when price returns to the middle of the Donchian channel.
# This avoids whipsaws in strong trends and focuses on mean reversion in ranging markets.
# Target: 20-40 trades/year on 4h to minimize fee drag while capturing range-bound moves.

name = "4h_Donchian_Breakout_Volume_Regime"
timeframe = "4h"
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

    # Donchian(20) on 4h
    period = 20
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    middle = (upper + lower) / 2.0

    # Chop regime filter (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10((highest_high - lowest_low) / (atr * 14)) / np.log10(14)
    chop = np.where((highest_high - lowest_low) > 0, chop, 50)  # avoid div by zero

    # Volume confirmation: volume > 1.5x 24-period average
    vol_avg_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(24, n):
        # Skip if any required value is NaN
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(middle[i]) or
            np.isnan(chop[i]) or np.isnan(vol_avg_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Break above upper band + volume chop + range market
            if (close[i] > upper[i] and
                volume[i] > vol_avg_24[i] * 1.5 and
                chop[i] > 61.8):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below lower band + volume chop + range market
            elif (close[i] < lower[i] and
                  volume[i] > vol_avg_24[i] * 1.5 and
                  chop[i] > 61.8):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Return to middle of channel
            if close[i] < middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Return to middle of channel
            if close[i] > middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals