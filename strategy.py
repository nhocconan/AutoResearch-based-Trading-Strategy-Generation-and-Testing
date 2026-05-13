#!/usr/bin/env python3
# 4H_WILLIAMS_ALLIGATOR_TREND_VOLUME
# Hypothesis: Williams Alligator (SMAs of median price) identifies trending regimes; entries occur on breakouts from Alligator's "lips" with volume confirmation, exits when price re-enters the Alligator's mouth. Works in bull/bear by capturing sustained moves.
# Uses 13/8/5 SMAs of (high+low)/2. Long when price > teeth + lips above teeth + volume spike; short when price < teeth + lips below teeth + volume spike. Exit when price crosses back through teeth.

name = "4H_WILLIAMS_ALLIGATOR_TREND_VOLUME"
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

    # Median price for Alligator
    median_price = (high + low) / 2.0

    # Williams Alligator: Jaw (13), Teeth (8), Lips (5) SMAs of median price
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values

    # Volume confirmation: volume > 1.7x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(13, n):
        # Skip if any required value is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price above teeth + lips above teeth + volume spike
            if (close[i] > teeth[i] and 
                lips[i] > teeth[i] and
                volume[i] > vol_avg_20[i] * 1.7):
                signals[i] = 0.25
                position = 1
            # SHORT: Price below teeth + lips below teeth + volume spike
            elif (close[i] < teeth[i] and 
                  lips[i] < teeth[i] and
                  volume[i] > vol_avg_20[i] * 1.7):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses back below teeth (Alligator waking)
            if close[i] < teeth[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses back above teeth (Alligator waking)
            if close[i] > teeth[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals