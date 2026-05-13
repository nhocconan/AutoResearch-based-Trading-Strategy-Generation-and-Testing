#!/usr/bin/env python3
# 6h_ElderRay_BullBearPower_1dTrend_Filter
# Hypothesis: Elder Ray (Bull/Bear Power) with 13-period EMA on 6h, filtered by 1d trend.
# Bull Power = High - EMA13, Bear Power = EMA13 - Low.
# Long when Bull Power > 0 and rising, Bear Power < 0 and falling, in uptrend (1d EMA50).
# Short when Bear Power < 0 and falling, Bull Power > 0 and rising, in downtrend (1d EMA50).
# Works in bull markets via long signals in uptrends, and bear markets via short signals in downtrends.
# Target: 20-50 trades/year per symbol to minimize fee drag.

name = "6h_ElderRay_BullBearPower_1dTrend_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values

    # EMA13 for Elder Ray
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values

    # Bull Power and Bear Power
    bull_power = high - ema13
    bear_power = ema13 - low

    # 1d trend filter: EMA50
    df_1d = get_htf_data(prices, '1d')
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(13, n):
        # Skip if any required value is NaN
        if (np.isnan(ema13[i]) or 
            np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or 
            np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Bull Power > 0 and rising, Bear Power < 0 and falling, in uptrend
            if (bull_power[i] > 0 and 
                i > 13 and bull_power[i] > bull_power[i-1] and 
                bear_power[i] < 0 and 
                i > 13 and bear_power[i] < bear_power[i-1] and 
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Bear Power < 0 and falling, Bull Power > 0 and rising, in downtrend
            elif (bear_power[i] > 0 and 
                  i > 13 and bear_power[i] > bear_power[i-1] and 
                  bull_power[i] < 0 and 
                  i > 13 and bull_power[i] < bull_power[i-1] and 
                  close[i] < ema50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bull Power <= 0 or trend turns down
            if bull_power[i] <= 0 or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bear Power <= 0 or trend turns up
            if bear_power[i] <= 0 or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals