#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Power with 12h EMA34 trend filter and volume confirmation.
- Elder Ray: Bull Power = high - EMA13, Bear Power = low - EMA13 (using 12h EMA13)
- Long: Bull Power > 0 + price > 12h EMA34 (uptrend) + volume > 2.0x 24-period avg
- Short: Bear Power < 0 + price < 12h EMA34 (downtrend) + volume > 2.0x 24-period avg
- Exit: Elder Power crosses zero (momentum shift) OR price crosses 12h EMA34 (trend exit)
- 12h EMA34 provides strong trend alignment to reduce whipsaws
- Volume confirmation ensures momentum validity
- Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe
- Discrete position sizing: ±0.25 to minimize fee churn
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
    
    # Volume confirmation: > 2.0x 24-period average (strict spike filter)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # Load 12h data ONCE before loop for EMA13 (Elder Ray) and EMA34 (trend)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate 12h EMA13 for Elder Ray Power
    ema_13_12h = pd.Series(close_12h).ewm(span=13, adjust=False, min_periods=13).mean().values
    # Calculate 12h EMA34 for trend filter
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 12h indicators to 6h timeframe
    ema_13_aligned = align_htf_to_ltf(prices, df_12h, ema_13_12h)
    ema_34_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Elder Ray Power: Bull Power = high - EMA13, Bear Power = low - EMA13
    bull_power = high_12h - ema_13_12h
    bear_power = low_12h - ema_13_12h
    bull_power_aligned = align_htf_to_ltf(prices, df_12h, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_12h, bear_power)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(24, 13, 34)  # Need 24 for volume MA, 13 for EMA13, 34 for EMA34
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(ema_13_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or 
            np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike confirmation (> 2.0x average)
        volume_spike = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: Bull Power > 0 (buying pressure) + price > 12h EMA34 (uptrend) + volume spike
            if volume_spike and bull_power_aligned[i] > 0 and close[i] > ema_34_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 (selling pressure) + price < 12h EMA34 (downtrend) + volume spike
            elif volume_spike and bear_power_aligned[i] < 0 and close[i] < ema_34_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bull Power <= 0 (buying pressure gone) OR price < 12h EMA34 (trend break)
            if bull_power_aligned[i] <= 0 or close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bear Power >= 0 (selling pressure gone) OR price > 12h EMA34 (trend break)
            if bear_power_aligned[i] >= 0 or close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_Power_12hEMA34_VolumeSpike"
timeframe = "6h"
leverage = 1.0