#!/usr/bin/env python3
# 12h_Alligator_Turn_WeeklyTrend_Filter
# Hypothesis: Williams Alligator signals on 12h timeframe filtered by weekly trend direction.
# Long when Alligator lines turn bullish (jaws < teeth < lips) during weekly uptrend.
# Short when Alligator lines turn bearish (jaws > teeth > lips) during weekly downtrend.
# Uses 1-week trend filter to avoid counter-trend whipsaws, targeting 15-30 trades/year per symbol.
# Designed to work in bull markets (trend-following) and bear markets (counter-trend bounces during weekly mean reversion).

name = "12h_Alligator_Turn_WeeklyTrend_Filter"
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

    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly EMA34 for trend direction
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)

    # Williams Alligator on 12h: SMMA(13,8), SMMA(8,5), SMMA(5,3)
    def smma(arr, period):
        result = np.full_like(arr, np.nan, dtype=np.float64)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (prev*(period-1) + current) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result

    jaws = smma(close, 13)  # Blue line: 13-period SMMA shifted 8 bars
    teeth = smma(close, 8)   # Red line: 8-period SMMA shifted 5 bars
    lips = smma(close, 5)    # Green line: 5-period SMMA shifted 3 bars

    # Shift the lines as per Alligator definition
    jaws = np.roll(jaws, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    # Set shifted values to NaN
    jaws[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(jaws[i]) or 
            np.isnan(teeth[i]) or np.isnan(lips[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Alligator lines turn bullish (jaws < teeth < lips) during weekly uptrend
            if (jaws[i] < teeth[i] < lips[i]) and (close[i] > ema_34_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Alligator lines turn bearish (jaws > teeth > lips) during weekly downtrend
            elif (jaws[i] > teeth[i] > lips[i]) and (close[i] < ema_34_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Alligator lines turn bearish or weekly trend turns down
            if (jaws[i] > teeth[i] > lips[i]) or (close[i] < ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Alligator lines turn bullish or weekly trend turns up
            if (jaws[i] < teeth[i] < lips[i]) or (close[i] > ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals