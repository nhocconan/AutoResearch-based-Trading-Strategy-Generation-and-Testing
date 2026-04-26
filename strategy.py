#!/usr/bin/env python3
"""
6h_ElderRay_1dTrend_VolumeConfirm
Hypothesis: Elder Ray (Bull/Bear Power) on 6h with 1d trend filter (price > EMA50) and volume confirmation (>2.0x EMA20 volume).
Enters long when Bull Power > 0 (close > EMA13) with bullish 1d trend and volume spike.
Enters short when Bear Power < 0 (close < EMA13) with bearish 1d trend and volume spike.
Exits when Elder Power reverses sign or volume drops.
Uses 6h timeframe with 1d HTF for trend. Designed for 50-150 total trades over 4 years (12-37/year) to avoid fee drag.
Uses discrete position sizing (0.25) to minimize churn. Works in both bull and bear markets by following 1d trend.
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
    
    # Calculate Elder Ray on 6h: EMA13, Bull Power = close - EMA13, Bear Power = EMA13 - close
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = close - ema_13
    bear_power = ema_13 - close  # positive when close < EMA13
    
    # Load 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: volume > 2.0 * 20-period EMA volume
    avg_volume = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (2.0 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 13-period EMA + 50-period 1d EMA)
    start_idx = max(13, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long logic: Bull Power > 0 + bullish 1d trend + volume spike
        if bull_power[i] > 0 and close[i] > ema_50_1d_aligned[i] and volume_spike[i]:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: Bear Power > 0 (close < EMA13) + bearish 1d trend + volume spike
        elif bear_power[i] > 0 and close[i] < ema_50_1d_aligned[i] and volume_spike[i]:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit: Elder Power reverses or volume normalizes
        elif position == 1 and (bull_power[i] <= 0 or not volume_spike[i]):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (bear_power[i] <= 0 or not volume_spike[i]):
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

name = "6h_ElderRay_1dTrend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0