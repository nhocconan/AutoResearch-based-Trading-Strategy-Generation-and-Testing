#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Power with 1d EMA34 trend filter and volume confirmation.
- Primary timeframe: 6h, HTF: 1d for trend filter and Elder Ray calculation
- Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
- Long: Bull Power > 0 AND Bear Power increasing (less negative) AND price > 1d EMA34 (uptrend) AND volume > 1.5x 20-period avg
- Short: Bear Power < 0 AND Bull Power decreasing (less positive) AND price < 1d EMA34 (downtrend) AND volume > 1.5x 20-period avg
- Exit: Opposite signal triggers (Bear Power >= 0 for long exit, Bull Power <= 0 for short exit)
- Uses Elder Ray to measure bull/bear power relative to trend, effective in both trending and ranging markets
- Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe
- Discrete position sizing: ±0.25 to minimize fee churn
- Works in bull markets (strong Bull Power with uptrend) and bear markets (strong Bear Power with downtrend)
- BTC/ETH focus: avoids SOL-only bias by requiring HTF trend alignment and volume confirmation
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume confirmation: > 1.5x 20-period average (volume spike filter)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d EMA13 for Elder Ray (shorter period for responsiveness)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # EMA13 for Elder Ray calculation
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power_1d = high_1d - ema_13_1d  # High - EMA13
    bear_power_1d = low_1d - ema_13_1d   # Low - EMA13
    
    # Align Elder Ray components to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 13, 34)  # Need 20 for volume MA, 13 for EMA13, 34 for EMA34
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or 
            np.isnan(ema_34_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike confirmation (> 1.5x average)
        volume_spike = volume[i] > 1.5 * vol_ma[i]
        
        # Elder Ray momentum: Bull Power increasing (less negative) or Bear Power increasing (less negative)
        # Compare with previous bar's values
        if i > 0:
            bull_power_prev = bull_power_aligned[i-1]
            bear_power_prev = bear_power_aligned[i-1]
            bull_power_increasing = bull_power_aligned[i] > bull_power_prev
            bear_power_increasing = bear_power_aligned[i] > bear_power_prev  # less negative = increasing
        else:
            bull_power_increasing = False
            bear_power_increasing = False
        
        if position == 0:
            # Long: Bull Power > 0 AND Bear Power increasing AND price > 1d EMA34 (uptrend) AND volume spike
            if (bull_power_aligned[i] > 0 and 
                bear_power_increasing and 
                close[i] > ema_34_aligned[i] and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 AND Bull Power decreasing AND price < 1d EMA34 (downtrend) AND volume spike
            elif (bear_power_aligned[i] < 0 and 
                  not bull_power_increasing and  # Bull Power decreasing (less positive or more negative)
                  close[i] < ema_34_aligned[i] and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bear Power >= 0 (bull power weakening)
            if bear_power_aligned[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bull Power <= 0 (bear power weakening)
            if bull_power_aligned[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_Power_1dEMA34_VolumeSpike"
timeframe = "6h"
leverage = 1.0