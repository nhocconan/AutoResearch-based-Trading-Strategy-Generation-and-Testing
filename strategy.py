#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Elder Ray (Bull/Bear Power) with EMA13 trend filter and volume confirmation.
# Bull Power = High - EMA13, Bear Power = Low - EMA13.
# Long when Bull Power > 0 and rising, price > EMA13, volume > 1.3x average.
# Short when Bear Power < 0 and falling, price < EMA13, volume > 1.3x average.
# Uses 1d EMA13 as trend filter on 6s timeframe for multi-timeframe alignment.
# Designed to work in bull (follow bull power) and bear (follow bear power) markets.
# Target: 60-120 total trades over 4 years (15-30/year) to avoid fee drag.

name = "6h_ElderRay_EMA13_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA13 and Elder Ray calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate EMA13 on 1d close
    ema13 = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high_1d - ema13
    bear_power = low_1d - ema13
    
    # Align 1d indicators to 6s timeframe
    ema13_aligned = align_htf_to_ltf(prices, df_1d, ema13)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Calculate 1-period change for Elder Ray slope (rising/falling)
    bull_power_rising = bull_power_aligned > np.roll(bull_power_aligned, 1)
    bear_power_falling = bear_power_aligned < np.roll(bear_power_aligned, 1)
    # Handle first element
    bull_power_rising[0] = False
    bear_power_falling[0] = False
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_bar = 0
    
    start_idx = 20  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema13_aligned[i]) or np.isnan(bull_power_aligned[i]) or
            np.isnan(bear_power_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull Power > 0 and rising, price > EMA13, volume spike
            if (bull_power_aligned[i] > 0 and bull_power_rising[i] and
                close[i] > ema13_aligned[i] and
                vol_ratio[i] > 1.3):
                signals[i] = 0.25
                position = 1
                entry_bar = i
            # Short: Bear Power < 0 and falling, price < EMA13, volume spike
            elif (bear_power_aligned[i] < 0 and bear_power_falling[i] and
                  close[i] < ema13_aligned[i] and
                  vol_ratio[i] > 1.3):
                signals[i] = -0.25
                position = -1
                entry_bar = i
        elif position == 1:
            # Long exit: Bull Power <= 0 or price <= EMA13
            if (bull_power_aligned[i] <= 0 or
                close[i] <= ema13_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bear Power >= 0 or price >= EMA13
            if (bear_power_aligned[i] >= 0 or
                close[i] >= ema13_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals