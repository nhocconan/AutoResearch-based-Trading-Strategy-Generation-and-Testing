#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA40 trend filter and volume confirmation.
# Bull Power = High - EMA40(1d), Bear Power = EMA40(1d) - Low.
# Long when Bull Power > 0 AND Bear Power < 0 (bullish momentum) AND 1d volume > 1.5x 20 EMA.
# Short when Bear Power > 0 AND Bull Power < 0 (bearish momentum) AND 1d volume > 1.5x 20 EMA.
# Uses 1d EMA40 for trend direction and volume for momentum confirmation.
# Designed for fewer trades (target: 15-25/year) to reduce fee drag and improve generalization.
# Works in both bull and bear markets by capturing momentum in the direction of higher timeframe trend.
name = "6h_ElderRay_1dEMA40_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for EMA40 and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA40 for trend
    ema40_1d = pd.Series(df_1d['close']).ewm(span=40, adjust=False, min_periods=40).mean().values
    
    # 1d volume > 1.5x 20 EMA for momentum confirmation
    vol_ema_20 = pd.Series(df_1d['volume']).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_strong = np.where(vol_ema_20 > 0, df_1d['volume'].values / vol_ema_20, 1.0) > 1.5
    vol_strong_aligned = align_htf_to_ltf(prices, df_1d, vol_strong)
    
    # Calculate Elder Ray components using 1d EMA40
    bull_power = high - ema40_1d[-1]  # Will be updated per bar via alignment
    bear_power = ema40_1d[-1] - low   # Will be updated per bar via alignment
    
    # Properly align Bull Power and Bear Power
    bull_power_raw = high - ema40_1d
    bear_power_raw = ema40_1d - low
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_raw)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_raw)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(vol_strong_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long condition: Bull Power > 0 AND Bear Power < 0 (bullish momentum) AND strong volume
            long_condition = (bull_power_aligned[i] > 0) and (bear_power_aligned[i] < 0) and vol_strong_aligned[i]
            # Short condition: Bear Power > 0 AND Bull Power < 0 (bearish momentum) AND strong volume
            short_condition = (bear_power_aligned[i] > 0) and (bull_power_aligned[i] < 0) and vol_strong_aligned[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: momentum deteriorates (Bear Power >= 0) or weak volume
            if (bear_power_aligned[i] >= 0) or not vol_strong_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: momentum deteriorates (Bull Power >= 0) or weak volume
            if (bull_power_aligned[i] >= 0) or not vol_strong_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals