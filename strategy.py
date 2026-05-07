#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with Elder Ray confirmation and volume spike.
# Uses Williams Alligator (Jaw/Teeth/Lips) from 1d data for trend direction,
# Elder Ray (Bull/Bear Power) for momentum confirmation, and volume spikes for entry timing.
# Designed to work in both bull and bear markets by following the 1d trend direction.
# Target: 15-35 trades/year per symbol to minimize fee drag.
name = "12h_WilliamsAlligator_ElderRay_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for Williams Alligator and Elder Ray
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Williams Alligator (13,8,5 SMAs with future shifts)
    jaw = pd.Series(df_1d['close']).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(df_1d['close']).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(df_1d['close']).rolling(window=5, min_periods=5).mean().shift(3).values
    
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = pd.Series(df_1d['close']).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = (df_1d['high'] - ema13).values
    bear_power = (df_1d['low'] - ema13).values
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # 12h volume average for spike detection
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = np.where(vol_ema > 0, volume / vol_ema, 1.0) > 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Alligator alignment: Lips > Teeth > Jaw = uptrend, Lips < Teeth < Jaw = downtrend
        uptrend = lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i]
        downtrend = lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i]
        
        if position == 0:
            # Long: Lips above Jaw (bullish alignment) + Bull Power positive + volume spike
            long_condition = (lips_aligned[i] > jaw_aligned[i]) and (bull_power_aligned[i] > 0) and vol_spike[i]
            # Short: Lips below Jaw (bearish alignment) + Bear Power negative + volume spike
            short_condition = (lips_aligned[i] < jaw_aligned[i]) and (bear_power_aligned[i] < 0) and vol_spike[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Lips cross below Jaw or Bull Power turns negative
            if (lips_aligned[i] < jaw_aligned[i]) or (bull_power_aligned[i] <= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Lips cross above Jaw or Bear Power turns positive
            if (lips_aligned[i] > jaw_aligned[i]) or (bear_power_aligned[i] >= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals