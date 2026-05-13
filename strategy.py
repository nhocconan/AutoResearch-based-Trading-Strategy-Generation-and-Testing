#!/usr/bin/env python3
# 6h_ElderRay_BullPower_1wTrend_Volume
# Hypothesis: Elder Ray Bull Power (close - EMA13) and Bear Power (EMA13 - low) on 6h
# combined with weekly trend filter (price > weekly EMA20) and volume confirmation.
# Long when Bull Power > 0 and rising, Bear Power < 0, price > weekly EMA20, volume spike.
# Short when Bear Power < 0 and falling, Bull Power > 0, price < weekly EMA20, volume spike.
# Designed to capture institutional buying/selling pressure with trend and volume confirmation.
# Target: 12-37 trades/year per symbol to minimize fee drag while maintaining edge in both bull and bear markets.

name = "6h_ElderRay_BullPower_1wTrend_Volume"
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

    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')

    # Calculate EMA13 for Elder Ray on 6h data
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values

    # Elder Ray components
    bull_power = close - ema13  # Close - EMA13
    bear_power = ema13 - low    # EMA13 - Low

    # Weekly EMA20 for trend filter
    ema20_1w = pd.Series(df_1w['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)

    # Volume confirmation: current volume > 2.0 x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):  # Start after sufficient warmup
        # Skip if any required value is NaN
        if (np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or 
            np.isnan(ema20_1w_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Bull Power positive and rising, Bear Power negative, above weekly EMA20, volume spike
            if (bull_power[i] > 0 and 
                i > 20 and bull_power[i] > bull_power[i-1] and  # Rising bull power
                bear_power[i] < 0 and
                close[i] > ema20_1w_aligned[i] and
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Bear Power negative and falling, Bull Power positive, below weekly EMA20, volume spike
            elif (bear_power[i] < 0 and 
                  i > 20 and bear_power[i] < bear_power[i-1] and  # Falling bear power
                  bull_power[i] > 0 and
                  close[i] < ema20_1w_aligned[i] and
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bull Power turns negative or Bear Power turns positive
            if bull_power[i] < 0 or bear_power[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bear Power turns positive or Bull Power turns negative
            if bear_power[i] > 0 or bull_power[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals