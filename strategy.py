#!/usr/bin/env python3
"""
12h_WilliamsAlligator_ElderRay
Hypothesis: On 12h timeframe, use Williams Alligator (JAW, TEETH, LIPS) for trend direction,
Elder Ray (Bull Power/Bear Power) for momentum confirmation, and 1-week high-low
range for volatility filter. Enter long when price > TEETH, Bull Power > 0, and price
above 1-week 50% retracement level; enter short when price < TEETH, Bear Power < 0,
and price below 1-week 50% retracement level. Uses Williams Alligator for trend,
Elder Ray for momentum, and weekly range for volatility filtering to capture trends
in both bull and bear markets while minimizing false signals.
"""

name = "12h_WilliamsAlligator_ElderRay"
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

    # Get 1-week data for volatility filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values

    # Williams Alligator (13,8,5 SMAs smoothed with 8,5,3 periods)
    # JAW: 13-period SMA, smoothed by 8 periods
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean()
    jaw = jaw.rolling(window=8, min_periods=8).mean().values
    # TEETH: 8-period SMA, smoothed by 5 periods
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean()
    teeth = teeth.rolling(window=5, min_periods=5).mean().values
    # LIPS: 5-period SMA, smoothed by 3 periods
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean()
    lips = lips.rolling(window=3, min_periods=3).mean().values

    # Elder Ray (13-period EMA for Bull/Bear Power)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13

    # 1-week high-low range for volatility filter
    # Calculate 50% retracement level of weekly range
    weekly_range = high_1w - low_1w
    weekly_mid = low_1w + (weekly_range * 0.5)
    weekly_mid_aligned = align_htf_to_ltf(prices, df_1w, weekly_mid)

    # Align Williams Alligator and Elder Ray to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, prices, jaw)  # Already 12h data
    teeth_aligned = align_htf_to_ltf(prices, prices, teeth)
    lips_aligned = align_htf_to_ltf(prices, prices, lips)
    bull_power_aligned = align_htf_to_ltf(prices, prices, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, prices, bear_power)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(13, n):  # Start after Alligator warmup
        # Skip if any required value is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or np.isnan(weekly_mid_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price > TEETH, Bull Power > 0, and price above weekly 50% level
            if (close[i] > teeth_aligned[i] and 
                bull_power_aligned[i] > 0 and 
                close[i] > weekly_mid_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price < TEETH, Bear Power < 0, and price below weekly 50% level
            elif (close[i] < teeth_aligned[i] and 
                  bear_power_aligned[i] < 0 and 
                  close[i] < weekly_mid_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price < TEETH OR Bull Power <= 0
            if close[i] < teeth_aligned[i] or bull_power_aligned[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price > TEETH OR Bear Power >= 0
            if close[i] > teeth_aligned[i] or bear_power_aligned[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals