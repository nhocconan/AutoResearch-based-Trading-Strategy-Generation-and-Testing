#!/usr/bin/env python3
"""
6h_ElderRay_ZeroLine_Cross
Hypothesis: Uses Elder Ray Index (Bull/Bear Power) with zero-line cross and EMA(13) trend filter on 6h timeframe.
Bull Power = High - EMA(13), Bear Power = Low - EMA(13).
Goes long when Bull Power crosses above zero with rising EMA(13), short when Bear Power crosses below zero with falling EMA(13).
Includes volume confirmation (volume > 1.5x 20-period average) to avoid false signals.
Designed for low-to-moderate trade frequency (~15-30/year) and works in both bull and bear markets by capturing momentum shifts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # EMA(13) for trend and power calculation
    ema13 = np.full(n, np.nan)
    k = 2 / (13 + 1)
    for i in range(13, n):
        if i == 13:
            ema13[i] = np.mean(close[0:14])
        else:
            ema13[i] = close[i] * k + ema13[i-1] * (1 - k)
    
    # Elder Ray: Bull Power and Bear Power
    bull_power = high - ema13
    bear_power = low - ema13
    
    # EMA13 slope (trend direction)
    ema13_slope = np.full(n, np.nan)
    for i in range(14, n):
        ema13_slope[i] = ema13[i] - ema13[i-1]
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 25  # Warmup
    
    for i in range(start_idx, n):
        if np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(ema13_slope[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Bull Power crosses above zero with rising EMA13 and volume spike
            if bull_power[i] > 0 and bull_power[i-1] <= 0 and ema13_slope[i] > 0 and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power crosses below zero with falling EMA13 and volume spike
            elif bear_power[i] < 0 and bear_power[i-1] >= 0 and ema13_slope[i] < 0 and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Bull Power crosses below zero or EMA13 turns down
            if bull_power[i] <= 0 or ema13_slope[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Bear Power crosses above zero or EMA13 turns up
            if bear_power[i] >= 0 or ema13_slope[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_ZeroLine_Cross"
timeframe = "6h"
leverage = 1.0