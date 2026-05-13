# !/usr/bin/env python3
"""
6h_Alligator_AllLines_ElderRay_Trend
Hypothesis: Combining Williams Alligator (JAWS/TEETH/LIPS) with Elder Ray (Bull/Bear Power) on 6h timeframe
provides robust trend detection with momentum confirmation, working in both bull and bear markets.
Alligator lines define trend direction (bullish when LIPS>TEETH>JAWS, bearish when LIPS<TEETH<JAWS).
Elder Ray confirms trend strength (Bull Power > 0 and rising for longs, Bear Power < 0 and falling for shorts).
Trades only when both indicators agree, reducing false signals and whipsaws.
Target: 15-35 trades/year per symbol.
"""

name = "6h_Alligator_AllLines_ElderRay_Trend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Williams Alligator: SMAs of median price (HL/2)
    median_price = (high + low) / 2.0
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Smooth Elder Ray for trend confirmation (3-period EMA)
    bull_power_smooth = pd.Series(bull_power).ewm(span=3, adjust=False, min_periods=3).mean().values
    bear_power_smooth = pd.Series(bear_power).ewm(span=3, adjust=False, min_periods=3).mean().values
    
    # Alligator trend conditions
    bullish_alligator = (lips > teeth) & (teeth > jaw)
    bearish_alligator = (lips < teeth) & (teeth < jaw)
    
    # Elder Ray trend conditions (require positive/negative and rising/falling)
    bullish_elder = (bull_power_smooth > 0) & (bull_power_smooth > np.roll(bull_power_smooth, 1))
    bearish_elder = (bear_power_smooth < 0) & (bear_power_smooth < np.roll(bear_power_smooth, 1))
    
    # Combine signals: both indicators must agree
    bullish = bullish_alligator & bullish_elder
    bearish = bearish_alligator & bearish_elder
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if position == 0:
            # LONG: bullish alignment from both indicators
            if bullish[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: bearish alignment from both indicators
            elif bearish[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: either indicator turns bearish
            if not (bullish_alligator[i] and bullish_elder[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: either indicator turns bullish
            if not (bearish_alligator[i] and bearish_elder[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals