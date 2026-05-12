#!/usr/bin/env python3
# 1d_WilliamsAlligator_ElderRay_1wTrend
# Hypothesis: Combines Williams Alligator (Jaw/Teeth/Lips) for trend direction,
# Elder Ray (Bull/Bear Power) for momentum strength, and weekly trend filter.
# Only takes trades when all three align, reducing false signals.
# Works in bull/bear by following higher timeframe trend. Targets 20-50 trades/year.

name = "1d_WilliamsAlligator_ElderRay_1wTrend"
timeframe = "1d"
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

    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values

    # Weekly trend filter: EMA50
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)

    # Williams Alligator (13,8,5) - Smoothed Medians
    jaw = pd.Series(high).rolling(window=13, min_periods=13).mean().rolling(window=8, min_periods=8).mean()
    teeth = pd.Series(low).rolling(window=8, min_periods=8).mean().rolling(window=5, min_periods=5).mean()
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().rolling(window=3, min_periods=3).mean()
    
    jaw = jaw.values
    teeth = teeth.values
    lips = lips.values

    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(13, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema_50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Williams Alligator alignment: Lips > Teeth > Jaw = Uptrend
        # Lips < Teeth < Jaw = Downtrend
        alligator_long = lips[i] > teeth[i] and teeth[i] > jaw[i]
        alligator_short = lips[i] < teeth[i] and teeth[i] < jaw[i]

        if position == 0:
            # LONG: Alligator uptrend + Bull Power positive + Weekly uptrend
            if alligator_long and bull_power[i] > 0 and close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Alligator downtrend + Bear Power negative + Weekly downtrend
            elif alligator_short and bear_power[i] < 0 and close[i] < ema_50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Alligator turns down OR Bear Power negative
            if not alligator_long or bear_power[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Alligator turns up OR Bull Power positive
            if not alligator_short or bull_power[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals