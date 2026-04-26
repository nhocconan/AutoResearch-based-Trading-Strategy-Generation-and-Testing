#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_1dTrend_VolumeConfirm
Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA34 trend filter and volume spike confirmation.
Elder Ray measures bull/bear power relative to EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13.
Long when Bull Power > 0 AND Bear Power improving (less negative) AND price > 1d EMA34 (uptrend) AND volume spike.
Short when Bear Power < 0 AND Bull Power deteriorating (less positive) AND price < 1d EMA34 (downtrend) AND volume spike.
Uses discrete position sizing (0.0, ±0.25) targeting 50-150 total trades over 4 years (12-37/year).
Designed to work in both bull and bear markets by aligning with higher timeframe trend and using volume confirmation.
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
    volume = prices['volume'].values
    
    # Calculate EMA13 for Elder Ray (primary timeframe)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13  # Bull Power = High - EMA13
    bear_power = low - ema_13   # Bear Power = Low - EMA13
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 2.0 * 20-period EMA volume
    avg_volume = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (2.0 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup
    start_idx = max(13, 34, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long logic: Bull Power > 0 AND Bear Power improving (current < previous) AND price > 1d EMA34 (uptrend) AND volume spike
        if (bull_power[i] > 0 and bear_power[i] < bear_power[i-1] and 
            close[i] > ema_34_1d_aligned[i] and volume_spike[i]):
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: Bear Power < 0 AND Bull Power deteriorating (current < previous) AND price < 1d EMA34 (downtrend) AND volume spike
        elif (bear_power[i] < 0 and bull_power[i] < bull_power[i-1] and 
              close[i] < ema_34_1d_aligned[i] and volume_spike[i]):
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit: Elder Ray divergence or trend failure
        elif position == 1 and (bull_power[i] <= 0 or close[i] < ema_34_1d_aligned[i]):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (bear_power[i] >= 0 or close[i] > ema_34_1d_aligned[i]):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "6h_ElderRay_BullBearPower_1dTrend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0