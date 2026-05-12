#!/usr/bin/env python3
"""
1d_Williams_Alligator_Elder_Ray_VolumeSpike
Hypothesis: Williams Alligator (3 SMAs) defines trend direction, Elder Ray (bull/bear power) measures momentum strength, and volume spike confirms conviction. Works in bull/bear by following Alligator alignment: bulls when jaw<teeth<lips, bears when jaw>teeth>lips. Uses 1d timeframe for low trade frequency (<25/year) to minimize fee drag. Entry when Elder Ray confirms Alligator direction with volume spike.
"""

name = "1d_Williams_Alligator_Elder_Ray_VolumeSpike"
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

    # Get 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')

    # Williams Alligator on 1d: Jaw(13,8), Teeth(8,5), Lips(5,3)
    # All values shifted forward by offset: Jaw+8, Teeth+5, Lips+3
    close_1d = close  # Already 1d timeframe
    jaw = pd.Series(close_1d).rolling(window=13, min_periods=13).mean().shift(8)
    teeth = pd.Series(close_1d).rolling(window=8, min_periods=8).mean().shift(5)
    lips = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().shift(3)

    # Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low
    ema_13 = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean()
    bull_power = high - ema_13.values
    bear_power = ema_13.values - low

    # 1w EMA34 trend filter (only trade in direction of weekly trend)
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)

    # Volume spike: >2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start after warmup
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Alligator bullish alignment (jaw<teeth<lips) + Bull Power > 0 + volume spike
            if (jaw[i] < teeth[i] and teeth[i] < lips[i] and 
                bull_power[i] > 0 and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Alligator bearish alignment (jaw>teeth>lips) + Bear Power > 0 + volume spike
            elif (jaw[i] > teeth[i] and teeth[i] > lips[i] and 
                  bear_power[i] > 0 and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Alligator turns bearish OR Bull Power <= 0
            if not (jaw[i] < teeth[i] and teeth[i] < lips[i]) or bull_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Alligator turns bullish OR Bear Power <= 0
            if not (jaw[i] > teeth[i] and teeth[i] > lips[i]) or bear_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals