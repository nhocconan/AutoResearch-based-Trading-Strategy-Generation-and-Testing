#!/usr/bin/env python3
# 12h_Williams_Alligator_Elder_Ray_Volume_Confirmation
# Hypothesis: Williams Alligator (Jaws, Teeth, Lips) identifies trend direction,
# Elder Ray (Bull/Bear Power) confirms momentum, and volume spike filters false signals.
# Works in both bull and bear markets by only trading in the direction of the Alligator alignment.
# Target: 12-30 trades/year on 12h timeframe to minimize fee drag.

name = "12h_Williams_Alligator_Elder_Ray_Volume_Confirmation"
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
    
    # Williams Alligator: SMAs of median price
    median_price = (high + low) / 2
    jaws_period, teeth_period, lips_period = 13, 8, 5
    jaws_shift, teeth_shift, lips_shift = 8, 5, 3
    
    jaws = pd.Series(median_price).rolling(window=jaws_period, min_periods=jaws_period).mean().shift(jaws_shift).values
    teeth = pd.Series(median_price).rolling(window=teeth_period, min_periods=teeth_period).mean().shift(teeth_shift).values
    lips = pd.Series(median_price).rolling(window=lips_period, min_periods=lips_period).mean().shift(lips_shift).values
    
    # Alligator alignment: jaws < teeth < lips = downtrend, jaws > teeth > lips = uptrend
    alligator_up = (jaws > teeth) & (teeth > lips)
    alligator_down = (jaws < teeth) & (teeth < lips)
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Volume confirmation: 1.5x 20-period average
    vol_ma = np.zeros_like(volume)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma[i] = vol_sum / 20
        else:
            vol_ma[i] = np.nan
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(jaws_period + jaws_shift, teeth_period + teeth_shift, lips_period + lips_shift, 13, 20) + 5
    
    for i in range(start_idx, n):
        if (np.isnan(jaws[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Alligator aligned up AND Bull Power > 0 AND volume confirmation
            if alligator_up[i] and (bull_power[i] > 0) and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Alligator aligned down AND Bear Power < 0 AND volume confirmation
            elif alligator_down[i] and (bear_power[i] < 0) and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Alligator alignment breaks down OR Bull Power becomes negative
            if not alligator_up[i] or (bull_power[i] <= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Alligator alignment breaks up OR Bear Power becomes positive
            if not alligator_down[i] or (bear_power[i] >= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals