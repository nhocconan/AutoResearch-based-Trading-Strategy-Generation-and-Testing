#!/usr/bin/env python3
# 4h_WilliamsAlligator_ElderRay_Trend
# Hypothesis: Combines Williams Alligator trend identification with Elder Ray bull/bear power to capture strong trends while avoiding chop.
# Williams Alligator (Jaw/Teeth/Lips) defines trend direction and strength via alignment.
# Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) confirms trend conviction.
# Entry requires aligned Alligator + strong Elder Ray signal + volume confirmation.
# Exit on Alligator reversal or Elder Ray divergence.
# Designed for 4h timeframe to target 20-40 trades/year (80-160 total) with low frequency to minimize fee drag.
# Works in bull/bear via trend-following logic and avoids chop via Alligator convergence/divergence.

name = "4h_WilliamsAlligator_ElderRay_Trend"
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

    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)

    close_1w = df_1w['close'].values

    # Calculate weekly EMA20 for trend filter
    ema_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)

    # Williams Alligator on 4h data (13,8,5 SMAs shifted)
    # Jaw = SMA(13, 8 shift), Teeth = SMA(8, 5 shift), Lips = SMA(5, 3 shift)
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3).values

    # Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = ema13 - low

    # Volume confirmation: current volume > 1.3 x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.3 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if data is not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema13[i]) or np.isnan(ema_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Williams Alligator alignment: Lips > Teeth > Jaw = uptrend, Lips < Teeth < Jaw = downtrend
        alligator_long = lips[i] > teeth[i] and teeth[i] > jaw[i]
        alligator_short = lips[i] < teeth[i] and teeth[i] < jaw[i]

        if position == 0:
            # LONG: Alligator aligned up + Bull Power positive + volume spike + weekly uptrend
            if alligator_long and bull_power[i] > 0 and volume_spike[i] and close[i] > ema_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Alligator aligned down + Bear Power positive + volume spike + weekly downtrend
            elif alligator_short and bear_power[i] > 0 and volume_spike[i] and close[i] < ema_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Alligator convergence (Lips < Teeth) or Bear Power positive or weekly trend down
            if (lips[i] < teeth[i]) or (bear_power[i] > 0) or (close[i] < ema_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Alligator convergence (Lips > Teeth) or Bull Power positive or weekly trend up
            if (lips[i] > teeth[i]) or (bull_power[i] > 0) or (close[i] > ema_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals