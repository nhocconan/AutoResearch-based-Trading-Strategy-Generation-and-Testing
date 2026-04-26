#!/usr/bin/env python3
"""
1d_WilliamsAlligator_ElderRay_HTFTrend
Hypothesis: On 1d timeframe, Williams Alligator (Jaw/Teeth/Lips) identifies market structure, Elder Ray (Bear/Bull Power) measures trend strength, and 1w EMA34 filter ensures alignment with higher timeframe trend. This combination captures strong trending moves while avoiding whipsaws in ranging markets. Works in both bull (buy dips) and bear (sell rallies) regimes by following the 1w trend.
"""

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
    
    # Load 1w data ONCE before loop for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Williams Alligator on 1d (Jaw=13, Teeth=8, Lips=5 SMAs of median price, shifted)
    median_price = (high + low) / 2
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().shift(8)
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().shift(5)
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().shift(3)
    
    jaw_aligned = jaw.values
    teeth_aligned = teeth.values
    lips_aligned = lips.values
    
    # Elder Ray on 1d: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(jaw_aligned[i]) or
            np.isnan(teeth_aligned[i]) or
            np.isnan(lips_aligned[i]) or
            np.isnan(bull_power[i]) or
            np.isnan(bear_power[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Williams Alligator alignment: Lips > Teeth > Jaw = uptrend, Lips < Teeth < Jaw = downtrend
        alligator_long = lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i]
        alligator_short = lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i]
        
        # Elder Ray confirmation: Bull Power > 0 and rising, Bear Power < 0 and falling
        bull_confirm = bull_power[i] > 0 and (i == start_idx or bull_power[i] > bull_power[i-1])
        bear_confirm = bear_power[i] < 0 and (i == start_idx or bear_power[i] < bear_power[i-1])
        
        # 1w trend filter
        uptrend_1w = close[i] > ema_34_1w_aligned[i]
        downtrend_1w = close[i] < ema_34_1w_aligned[i]
        
        # Long logic: Alligator aligned up + Elder Ray bullish + 1w uptrend
        if alligator_long and bull_confirm and uptrend_1w:
            if position != 1:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        # Short logic: Alligator aligned down + Elder Ray bearish + 1w downtrend
        elif alligator_short and bear_confirm and downtrend_1w:
            if position != -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = -0.25
        # Exit conditions: loss of Alligator alignment or Elder Ray divergence
        elif position == 1 and (not alligator_long or not bull_confirm):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (not alligator_short or not bear_confirm):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WilliamsAlligator_ElderRay_HTFTrend"
timeframe = "1d"
leverage = 1.0