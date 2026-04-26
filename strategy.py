#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_1wTrend_VolumeConfirm
Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1w trend filter and volume confirmation.
Elder Ray measures bull/bear strength relative to EMA13. Long when Bull Power > 0 and Bear Power < 0 (bullish momentum),
short when Bear Power > 0 and Bull Power < 0 (bearish momentum), filtered by 1w EMA34 trend.
Volume spike confirms institutional participation. Works in bull/bear markets by aligning with higher timeframe trend.
Target: 50-150 total trades over 4 years (12-37/year) with discrete position sizing (0.0, ±0.25).
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
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA34 for trend filter
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Elder Ray components: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Volume confirmation: volume > 2.0 * 20-period EMA volume
    avg_volume = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (2.0 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup
    start_idx = max(34, 13, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long logic: Bull Power > 0 AND Bear Power < 0 (bullish momentum) + price > 1w EMA34 (uptrend) + volume spike
        if bull_power[i] > 0 and bear_power[i] < 0 and close[i] > ema_34_1w_aligned[i] and volume_spike[i]:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: Bear Power > 0 AND Bull Power < 0 (bearish momentum) + price < 1w EMA34 (downtrend) + volume spike
        elif bear_power[i] > 0 and bull_power[i] < 0 and close[i] < ema_34_1w_aligned[i] and volume_spike[i]:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit: momentum divergence (Bull Power <= 0 for longs, Bear Power <= 0 for shorts)
        elif position == 1 and bull_power[i] <= 0:
            signals[i] = 0.0
            position = 0
        elif position == -1 and bear_power[i] <= 0:
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

name = "6h_ElderRay_BullBearPower_1wTrend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0