#!/usr/bin/env python3
# 12h_WilliamsAlligator_ElderRay_Trend_Volume
# Hypothesis: Williams Alligator (13/8/5 SMAs) defines trend direction, Elder Ray (bull/bear power) confirms momentum strength, and volume spike filters weak moves. Works in bull/bear by using 1w trend filter and requiring volume confirmation. Targets 12-37 trades/year on 12h timeframe.

name = "12h_WilliamsAlligator_ElderRay_Trend_Volume"
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

    # Calculate ATR for volatility normalization (used in Elder Ray)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values

    # 1w EMA34 for trend filter (load once, align)
    df_1w = get_htf_data(prices, '1w')
    ema34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)

    # Williams Alligator: Jaw(13), Teeth(8), Lips(5) SMAs
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().values

    # Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = ema13 - low

    # Volume confirmation: volume > 1.8x 30-period average
    vol_avg_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):
        # Skip if any required value is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema34_1w_aligned[i]) or np.isnan(vol_avg_30[i])):
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
            # LONG: Alligator uptrend + Bull Power > 0 + volume spike + 1w uptrend
            if (alligator_long and 
                bull_power[i] > 0 and
                volume[i] > vol_avg_30[i] * 1.8 and
                close[i] > ema34_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Alligator downtrend + Bear Power > 0 + volume spike + 1w downtrend
            elif (alligator_short and 
                  bear_power[i] > 0 and
                  volume[i] > vol_avg_30[i] * 1.8 and
                  close[i] < ema34_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Alligator turns down OR Elder Ray weakens
            if not alligator_long or bull_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Alligator turns up OR Elder Ray weakens
            if not alligator_short or bear_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals