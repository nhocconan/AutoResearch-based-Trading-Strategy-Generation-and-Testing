#!/usr/bin/env python3
"""
6h Williams Alligator + Elder Ray Power Strategy
Hypothesis: Williams Alligator (JAW/TEETH/LIPS) identifies trend phases (sleeping/awakening/eating). Elder Ray (Bull/Bear Power) measures trend strength relative to EMA13. Combining both filters out weak signals: enter only when Alligator is 'eating' (trending) AND Elder Power confirms direction. Works in bull/bear by requiring trend alignment. Discrete sizing (0.0, ±0.25) minimizes fee churn. Target: 15-30 trades/year on 6h.
"""

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
    volume = prices['volume'].values
    
    # Get 1d data for EMA13 trend filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Calculate 1d EMA13 for Elder Ray
    ema_13_1d = pd.Series(df_1d['close'].values).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    
    # Williams Alligator on 6h: JAW(13,8), TEETH(8,5), LIPS(5,3)
    # JAW: 13-period SMMA, smoothed 8 periods ahead
    jaw_raw = pd.Series(high).rolling(window=13, min_periods=13).mean().values
    jaw = pd.Series(jaw_raw).rolling(window=8, min_periods=8).mean().values
    
    # TEETH: 8-period SMMA, smoothed 5 periods ahead
    teeth_raw = pd.Series(low).rolling(window=8, min_periods=8).mean().values
    teeth = pd.Series(teeth_raw).rolling(window=5, min_periods=5).mean().values
    
    # LIPS: 5-period SMMA, smoothed 3 periods ahead
    lips_raw = pd.Series(close).rolling(window=5, min_periods=5).mean().values
    lips = pd.Series(lips_raw).rolling(window=3, min_periods=3).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13_1d_aligned
    bear_power = low - ema_13_1d_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Alligator calculations
    start_idx = 13 + 8  # JAW needs 13+8=21
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            signals[i] = 0.0
            continue
        
        # Alligator sleeping condition: all lines intertwined (JAW, TEETH, LIPS close)
        jaw_teeth_diff = abs(jaw[i] - teeth[i])
        teeth_lips_diff = abs(teeth[i] - lips[i])
        lips_jaw_diff = abs(lips[i] - jaw[i])
        max_diff = max(jaw_teeth_diff, teeth_lips_diff, lips_jaw_diff)
        atr_approx = pd.Series(high - low).rolling(window=14, min_periods=14).mean().values[i]
        sleeping = max_diff < (0.1 * atr_approx) if not np.isnan(atr_approx) else False
        
        # Alligator eating condition: lines separated in correct order
        # Uptrend: LIPS > TEETH > JAW
        # Downtrend: JAW > TEETH > LIPS
        eating_up = (lips[i] > teeth[i]) and (teeth[i] > jaw[i])
        eating_down = (jaw[i] > teeth[i]) and (teeth[i] > lips[i])
        
        if position == 0:
            # Look for entry signals
            # Long: Alligator eating up AND Bull Power > 0 (strong bullish)
            # Short: Alligator eating down AND Bear Power < 0 (strong bearish)
            long_entry = eating_up and (bull_power[i] > 0)
            short_entry = eating_down and (bear_power[i] < 0)
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: Alligator starts sleeping OR eating down OR Bull Power <= 0
            if sleeping or eating_down or (bull_power[i] <= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: Alligator starts sleeping OR eating up OR Bear Power >= 0
            if sleeping or eating_up or (bear_power[i] >= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsAlligator_ElderRay_Power"
timeframe = "6h"
leverage = 1.0