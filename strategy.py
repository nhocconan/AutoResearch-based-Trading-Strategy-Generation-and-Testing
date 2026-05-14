#!/usr/bin/env python3
# Hypothesis: 6h Williams Alligator + Elder Ray combination with 1w trend filter.
# Long when: Alligator jaws < teeth < lips (bullish alignment) AND Bull Power > 0 AND 1w close > 1w EMA20 (uptrend).
# Short when: Alligator jaws > teeth > lips (bearish alignment) AND Bear Power < 0 AND 1w close < 1w EMA20 (downtrend).
# Exit when Alligator alignment breaks (jaws-teeth-lips not in proper order) OR power crosses zero.
# Uses discrete position sizing (0.25) to limit fee churn. Designed for 6h timeframe with strict entry conditions.
# Target: 50-150 total trades over 4 years (12-37/year) for 6h.
# Alligator filters whipsaws, Elder Ray measures power, 1w EMA20 ensures higher timeframe trend alignment.

name = "6h_WilliamsAlligator_ElderRay_1wTrendFilter_v1"
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
    
    # Calculate Williams Alligator (SMMA with offsets)
    def smma(values, period):
        result = np.full_like(values, np.nan)
        if len(values) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(values[:period])
        # Subsequent values: SMMA = (PREV * (period-1) + CURRENT) / period
        for i in range(period, len(values)):
            if not np.isnan(result[i-1]):
                result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    # Alligator lines: Jaw (13,8), Teeth (8,5), Lips (5,3)
    jaw = smma((high + low) / 2, 13)
    jaw = np.roll(jaw, 8)  # offset by 8 bars
    teeth = smma((high + low) / 2, 8)
    teeth = np.roll(teeth, 5)  # offset by 5 bars
    lips = smma((high + low) / 2, 5)
    lips = np.roll(lips, 3)  # offset by 3 bars
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Calculate 1w EMA20 for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Bullish Alligator alignment AND Bull Power > 0 AND 1w close > 1w EMA20
            if (jaw[i] < teeth[i] and teeth[i] < lips[i] and  # jaws < teeth < lips
                bull_power[i] > 0 and
                close[i] > ema20_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Bearish Alligator alignment AND Bear Power < 0 AND 1w close < 1w EMA20
            elif (jaw[i] > teeth[i] and teeth[i] > lips[i] and  # jaws > teeth > lips
                  bear_power[i] < 0 and
                  close[i] < ema20_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Alligator alignment breaks OR Bull Power <= 0
            if not (jaw[i] < teeth[i] and teeth[i] < lips[i]) or bull_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Alligator alignment breaks OR Bear Power >= 0
            if not (jaw[i] > teeth[i] and teeth[i] > lips[i]) or bear_power[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals