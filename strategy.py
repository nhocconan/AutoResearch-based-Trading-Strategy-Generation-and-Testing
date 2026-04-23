#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Index with 1d EMA34 trend filter and volume confirmation.
- Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
- Long: Bull Power > 0 AND Bear Power rising (less negative) + price > 1d EMA34 + volume > 2x 20-period avg
- Short: Bear Power < 0 AND Bull Power falling (less positive) + price < 1d EMA34 + volume > 2x 20-period avg
- Exit: Opposite Elder Ray signal (Bull Power < 0 for long exit, Bear Power > 0 for short exit)
- Uses 1d EMA34 for trend filter, 6h Elder Ray for momentum/strength, volume for confirmation
- Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe
- Discrete position sizing: ±0.25 to minimize fee churn
- Works in bull markets (strong Bull Power) and bear markets (strong Bear Power)
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
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate EMA13 for Elder Ray (on 6h data)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # High - EMA13
    bear_power = low - ema_13   # Low - EMA13
    
    # Smooth Elder Ray signals to reduce noise (2-period EMA)
    bull_power_smooth = pd.Series(bull_power).ewm(span=2, adjust=False, min_periods=2).mean().values
    bear_power_smooth = pd.Series(bear_power).ewm(span=2, adjust=False, min_periods=2).mean().values
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 13, 34)  # Volume MA, EMA13, EMA34
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(bull_power_smooth[i]) or
            np.isnan(bear_power_smooth[i]) or
            np.isnan(ema_34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        # Elder Ray momentum: rising Bull Power or falling Bear Power
        bull_power_rising = i > start_idx and bull_power_smooth[i] > bull_power_smooth[i-1]
        bear_power_falling = i > start_idx and bear_power_smooth[i] < bear_power_smooth[i-1]
        
        if position == 0:
            # Long: Bull Power > 0 AND rising + bullish trend + volume confirmation
            if (bull_power_smooth[i] > 0 and 
                bull_power_rising and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 AND falling + bearish trend + volume confirmation
            elif (bear_power_smooth[i] < 0 and 
                  bear_power_falling and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bull Power < 0 (momentum weakening)
            if bull_power_smooth[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bear Power > 0 (momentum weakening)
            if bear_power_smooth[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_EMA13_Trend_Volume"
timeframe = "6h"
leverage = 1.0