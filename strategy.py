#!/usr/bin/env python3
"""
6h_ElderRay_ZeroCross_1dTrend_v1
Hypothesis: 6h Elder Ray (Bull/Bear Power) zero-cross with 1d EMA34 trend filter and volume confirmation.
- Bull Power = High - EMA13, Bear Power = Low - EMA13
- Long when Bull Power crosses above zero AND 1d uptrend AND volume spike
- Short when Bear Power crosses below zero AND 1d downtrend AND volume spike
- EMA13 acts as dynamic equilibrium; cross indicates momentum shift
- 1d EMA34 trend filter reduces whipsaw in bear markets and captures major moves
- Volume spike (2.0x 20-period average) confirms institutional participation
- Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate EMA13 for Elder Ray
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13  # Bull Power: High - EMA13
    bear_power = low - ema13   # Bear Power: Low - EMA13
    
    # Zero-cross detection
    bull_power_prev = np.roll(bull_power, 1)
    bear_power_prev = np.roll(bear_power, 1)
    bull_power_prev[0] = np.nan
    bear_power_prev[0] = np.nan
    
    bull_power_up = (bull_power > 0) & (bull_power_prev <= 0)  # Cross above zero
    bear_power_down = (bear_power < 0) & (bear_power_prev >= 0)  # Cross below zero
    
    # Calculate 1d EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate volume spike (20-period volume average on 6h)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 2.0)  # Volume at least 2.0x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 13 for EMA13, 20 for volume MA)
    start_idx = max(13, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(bull_power_up[i]) or np.isnan(bear_power_down[i]) or
            np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Elder Ray zero-cross conditions with volume confirmation and trend filter
        if position == 0:
            # Long: Bull Power crosses above zero AND 1d uptrend AND volume spike
            if bull_power_up[i] and close[i] > ema34_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power crosses below zero AND 1d downtrend AND volume spike
            elif bear_power_down[i] and close[i] < ema34_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Bear Power crosses below zero OR 1d trend turns down
            if bear_power_down[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Bull Power crosses above zero OR 1d trend turns up
            if bull_power_up[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ElderRay_ZeroCross_1dTrend_v1"
timeframe = "6h"
leverage = 1.0