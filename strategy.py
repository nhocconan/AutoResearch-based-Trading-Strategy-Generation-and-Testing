#!/usr/bin/env python3
# 6h_ElderRay_BullBearPower_1dTrend_Filter
# Hypothesis: Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) measures bull/bear strength.
# Long when Bull Power > 0 and Bear Power < 0 (bullish divergence) with 1d EMA50 uptrend.
# Short when Bear Power > 0 and Bull Power < 0 (bearish divergence) with 1d EMA50 downtrend.
# Uses volume confirmation to avoid false signals. Works in bull via bullish divergence + uptrend,
# and in bear via bearish divergence + downtrend. Target: 15-30 trades/year.

name = "6h_ElderRay_BullBearPower_1dTrend_Filter"
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

    # EMA13 for Elder Ray calculation
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values

    # Bull Power = High - EMA13, Bear Power = EMA13 - Low
    bull_power = high - ema13
    bear_power = ema13 - low

    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Volume filter: >1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Bull Power > 0 and Bear Power < 0 (bullish divergence) + price > EMA13 + 1d uptrend + volume spike
            if (bull_power[i] > 0 and bear_power[i] < 0 and 
                close[i] > ema13[i] and
                ema50_1d_aligned[i] > ema50_1d_aligned[i-1] and  # 1d EMA50 rising
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Bear Power > 0 and Bull Power < 0 (bearish divergence) + price < EMA13 + 1d downtrend + volume spike
            elif (bear_power[i] > 0 and bull_power[i] < 0 and 
                  close[i] < ema13[i] and
                  ema50_1d_aligned[i] < ema50_1d_aligned[i-1] and  # 1d EMA50 falling
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bearish divergence or price re-enters EMA13 or 1d trend turns down
            if (bear_power[i] > 0 or bull_power[i] < 0 or 
                close[i] < ema13[i] or
                ema50_1d_aligned[i] < ema50_1d_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bullish divergence or price re-enters EMA13 or 1d trend turns up
            if (bull_power[i] > 0 or bear_power[i] < 0 or 
                close[i] > ema13[i] or
                ema50_1d_aligned[i] > ema50_1d_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals