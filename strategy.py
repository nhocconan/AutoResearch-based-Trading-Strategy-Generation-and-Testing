#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Elder Ray (Bull/Bear Power) with EMA13 filter and volume confirmation.
# Long when 1d Bull Power > 0, EMA13 rising, and volume > 1.5x average.
# Short when 1d Bear Power < 0, EMA13 falling, and volume > 1.5x average.
# Works in bull (Bull Power positive) and bear (Bear Power negative) markets by measuring power relative to EMA.
# Target: 50-150 total trades over 4 years (12-37/year) to balance opportunity and fee drag.

name = "6h_ElderRay_EMA13_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Elder Ray and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d EMA13
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema13_rising = ema13_1d > np.roll(ema13_1d, 1)
    ema13_falling = ema13_1d < np.roll(ema13_1d, 1)
    ema13_rising = np.where(np.isnan(ema13_rising), False, ema13_rising)
    ema13_falling = np.where(np.isnan(ema13_falling), False, ema13_falling)
    
    # 1d Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high_1d - ema13_1d
    bear_power = low_1d - ema13_1d
    
    # Align 1d indicators to 6h
    ema13_rising_aligned = align_htf_to_ltf(prices, df_1d, ema13_rising.astype(float))
    ema13_falling_aligned = align_htf_to_ltf(prices, df_1d, ema13_falling.astype(float))
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 13  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema13_rising_aligned[i]) or np.isnan(ema13_falling_aligned[i]) or
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull Power > 0, EMA13 rising, volume spike
            if (bull_power_aligned[i] > 0 and
                ema13_rising_aligned[i] and
                vol_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0, EMA13 falling, volume spike
            elif (bear_power_aligned[i] < 0 and
                  ema13_falling_aligned[i] and
                  vol_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bull Power <= 0 or EMA13 not rising
            if (bull_power_aligned[i] <= 0 or not ema13_rising_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bear Power >= 0 or EMA13 not falling
            if (bear_power_aligned[i] >= 0 or not ema13_falling_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals