#!/usr/bin/env python3
"""
6h_1d_Alligator_ElderRay
Hypothesis: Combine Williams Alligator (trend detection) with Elder Ray (bull/bear power) on 1d timeframe.
Enter long when price > Alligator's Jaw AND Bull Power > 0 with rising 1d momentum.
Enter short when price < Alligator's Jaw AND Bear Power < 0 with falling 1d momentum.
Use 6h timeframe for entries with 0.25 position sizing.
Designed to catch sustained trends while avoiding sideways chop. Works in bull via trend following,
in bear via short signals during distribution phases.
Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_Alligator_ElderRay"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === 1D ALLIGATOR (JAW, TEETH, LIPS) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    median_1d = (df_1d['high'].values + df_1d['low'].values) / 2
    
    # Jaw (13-period SMMA, 8 bars ahead)
    jaw = pd.Series(median_1d).rolling(window=13, min_periods=13).mean().values
    jaw = np.roll(jaw, 8)  # shift forward 8 bars
    jaw[:8] = np.nan
    
    # Teeth (8-period SMMA, 5 bars ahead)
    teeth = pd.Series(median_1d).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth, 5)  # shift forward 5 bars
    teeth[:5] = np.nan
    
    # Lips (5-period SMMA, 3 bars ahead)
    lips = pd.Series(median_1d).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips, 3)  # shift forward 3 bars
    lips[:3] = np.nan
    
    # Align Alligator lines to 6h
    jaw_6h = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_6h = align_htf_to_ltf(prices, df_1d, teeth)
    lips_6h = align_htf_to_ltf(prices, df_1d, lips)
    
    # === 1D ELDER RAY (BULL/BEAR POWER) ===
    # Bull Power = High - EMA13
    # Bear Power = Low - EMA13
    ema13_1d = pd.Series(df_1d['close'].values).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = df_1d['high'].values - ema13_1d
    bear_power = df_1d['low'].values - ema13_1d
    
    # Align Elder Ray to 6h
    bull_power_6h = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_6h = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # === 1D MOMENTUM (ROC 3-period) ===
    roc3_1d = pd.Series(df_1d['close'].values).pct_change(periods=3).values * 100
    roc3_6h = align_htf_to_ltf(prices, df_1d, roc3_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if not ready
        if (np.isnan(jaw_6h[i]) or np.isnan(bull_power_6h[i]) or np.isnan(bear_power_6h[i]) or 
            np.isnan(roc3_6h[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Long conditions:
        # 1. Price above Alligator Jaw (uptrend)
        # 2. Bull Power positive (bulls in control)
        # 3. Rising 1d momentum (ROC3 > 0)
        long_signal = (close[i] > jaw_6h[i] and 
                      bull_power_6h[i] > 0 and 
                      roc3_6h[i] > 0)
        
        # Short conditions:
        # 1. Price below Alligator Jaw (downtrend)
        # 2. Bear Power negative (bears in control)
        # 3. Falling 1d momentum (ROC3 < 0)
        short_signal = (close[i] < jaw_6h[i] and 
                       bear_power_6h[i] < 0 and 
                       roc3_6h[i] < 0)
        
        # Exit: price crosses Teeth (trend weakening) or Elder Ray diverges
        exit_long = (position == 1 and 
                    (close[i] < teeth_6h[i] or bull_power_6h[i] < 0))
        exit_short = (position == -1 and 
                     (close[i] > teeth_6h[i] or bear_power_6h[i] > 0))
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals