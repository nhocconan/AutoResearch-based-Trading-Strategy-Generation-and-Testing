#!/usr/bin/env python3
# 12h_Williams_Alligator_ElderRay_1wTrend
# Hypothesis: Combines Williams Alligator (trend identification) with Elder Ray (bull/bear power) on 12h timeframe, filtered by 1w trend direction. 
# Alligator identifies trend via SMAs (Jaw/Teeth/Lips), Elder Ray measures bull/bear power via EMA13, and 1w EMA8 filters for long-term trend.
# Designed to work in both bull and bear markets by only taking trades in direction of higher timeframe trend, reducing whipsaws.

name = "12h_Williams_Alligator_ElderRay_1wTrend"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(series, period):
    """Calculate Exponential Moving Average."""
    return pd.Series(series).ewm(span=period, adjust=False, min_periods=period).mean().values

def sma(series, period):
    """Calculate Simple Moving Average."""
    return pd.Series(series).rolling(window=period, min_periods=period).mean().values

def williams_alligator(high, low, close, jaw_period=13, teeth_period=8, lips_period=5):
    """Calculate Williams Alligator (Jaw, Teeth, Lips)."""
    median_price = (high + low) / 2
    jaw = sma(median_price, jaw_period)
    teeth = sma(median_price, teeth_period)
    lips = sma(median_price, lips_period)
    return jaw, teeth, lips

def elder_ray(high, low, close, ema_period=13):
    """Calculate Elder Ray (Bull Power, Bear Power)."""
    ema_val = ema(close, ema_period)
    bull_power = high - ema_val
    bear_power = low - ema_val
    return bull_power, bear_power

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA8 for trend filter
    ema_8_1w = ema(df_1w['close'], 8)
    ema_8_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_8_1w)

    # Calculate Williams Alligator on 12h
    jaw, teeth, lips = williams_alligator(high, low, close, 13, 8, 5)
    
    # Calculate Elder Ray on 12h
    bull_power, bear_power = elder_ray(high, low, close, 13)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(ema_8_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Alligator alignment: Lips > Teeth > Jaw = uptrend, Lips < Teeth < Jaw = downtrend
        alligator_long = lips[i] > teeth[i] > jaw[i]
        alligator_short = lips[i] < teeth[i] < jaw[i]
        
        # Elder Ray: Bull Power > 0 and rising, Bear Power < 0 and falling
        bull_strong = bull_power[i] > 0 and (i == 0 or bull_power[i] > bull_power[i-1])
        bear_strong = bear_power[i] < 0 and (i == 0 or bear_power[i] < bear_power[i-1])

        if position == 0:
            # LONG: Alligator uptrend + Bull Power positive/rising + 1w uptrend
            if (alligator_long and bull_strong and close[i] > ema_8_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Alligator downtrend + Bear Power negative/falling + 1w downtrend
            elif (alligator_short and bear_strong and close[i] < ema_8_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Alligator trend weakness or Bear Power becomes strong
            if not alligator_long or bear_strong:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Alligator trend weakness or Bull Power becomes strong
            if not alligator_short or bull_strong:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals