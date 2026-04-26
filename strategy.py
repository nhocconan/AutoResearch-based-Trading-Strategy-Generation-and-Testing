#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_1dTrend_VolumeSpike
Hypothesis: Use 6h timeframe with Elder Ray (Bull Power/Bear Power) combined with 1d EMA trend filter and volume confirmation.
Long when: Bull Power > 0, Bear Power < 0 (bullish momentum), price > 1d EMA50 (uptrend), volume > 1.5 * avg volume.
Short when: Bull Power < 0, Bear Power > 0 (bearish momentum), price < 1d EMA50 (downtrend), volume > 1.5 * avg volume.
Exit when: momentum divergence (Bull Power turns negative for long, Bear Power turns positive for short) or opposite extreme.
Uses discrete 0.25 position size to limit fee drag. Targets 50-150 total trades over 4 years.
Works in both bull and bear markets via trend-adaptive momentum signals.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13 (using 6h EMA13)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Volume spike: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 13 for EMA13, 20 for volume avg, 50 for 1d EMA
    start_idx = max(13, 20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        size = 0.25  # Fixed position size
        
        if position == 0:
            # Flat - look for entry with trend and volume confirmation
            # Long: Bull Power > 0, Bear Power < 0 (bullish momentum), price > 1d EMA50 (uptrend), volume spike
            long_entry = (bull_power[i] > 0) and \
                       (bear_power[i] < 0) and \
                       (close_val > ema_50_1d_aligned[i]) and \
                       volume_spike[i]
            # Short: Bull Power < 0, Bear Power > 0 (bearish momentum), price < 1d EMA50 (downtrend), volume spike
            short_entry = (bull_power[i] < 0) and \
                        (bear_power[i] > 0) and \
                        (close_val < ema_50_1d_aligned[i]) and \
                        volume_spike[i]
            
            if long_entry:
                signals[i] = size
                position = 1
            elif short_entry:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit when bullish momentum fades (Bull Power turns negative) or bearish extreme
            if (bull_power[i] <= 0) or (bear_power[i] >= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when bearish momentum fades (Bear Power turns positive) or bullish extreme
            if (bull_power[i] >= 0) or (bear_power[i] <= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_ElderRay_BullBearPower_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0