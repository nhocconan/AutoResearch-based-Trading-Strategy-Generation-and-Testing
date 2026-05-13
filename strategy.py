#!/usr/bin/env python3
# 6h_ElderRay_BullBearPower_1dTrend_Filter
# Hypothesis: Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) captures bull/bear strength.
# Combined with 1d EMA50 trend filter to align with higher timeframe direction.
# In bull markets (price > 1d EMA50), take long when Bull Power > 0 and rising.
# In bear markets (price < 1d EMA50), take short when Bear Power > 0 and rising.
# Uses EMA13 for sensitivity and avoids whipsaws. Target: 15-30 trades per year per symbol.

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
    volume = prices['volume'].values

    # EMA13 for Elder Ray calculation
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values

    # Bull Power = High - EMA13, Bear Power = EMA13 - Low
    bull_power = high - ema13
    bear_power = ema13 - low

    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(13, n):
        # Skip if any required value is NaN
        if np.isnan(ema50_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Determine trend from 1d EMA50
        uptrend = close[i] > ema50_1d_aligned[i]
        downtrend = close[i] < ema50_1d_aligned[i]

        if position == 0:
            # LONG: Uptrend + Bull Power positive and rising (bullish momentum)
            if uptrend and bull_power[i] > 0 and bull_power[i] > bull_power[i-1]:
                signals[i] = 0.25
                position = 1
            # SHORT: Downtrend + Bear Power positive and rising (bearish momentum)
            elif downtrend and bear_power[i] > 0 and bear_power[i] > bear_power[i-1]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Trend reversal or Bull Power weakening
            if not uptrend or bull_power[i] <= 0 or bull_power[i] < bull_power[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Trend reversal or Bear Power weakening
            if not downtrend or bear_power[i] <= 0 or bear_power[i] < bear_power[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals