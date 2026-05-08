#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d trend filter and volume confirmation.
# Bull Power = High - EMA13(1d), Bear Power = EMA13(1d) - Low.
# Long when Bull Power > 0 AND 1d EMA13 rising AND volume > 1.5x 20-period average.
# Short when Bear Power > 0 AND 1d EMA13 falling AND volume > 1.5x 20-period average.
# Exit when power crosses zero (Bull Power < 0 for long exit, Bear Power < 0 for short exit).
# Elder Ray measures bull/bear strength relative to trend. EMA13 filters higher timeframe trend.
# Volume spike confirms institutional participation. Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_ElderRay_1dEMA13_Volume"
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
    
    # 1d data for EMA13 calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d EMA13 for trend filter
    ema13_1d = pd.Series(df_1d['close']).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema13_1d)
    
    # 1d EMA13 direction
    ema13_rising = np.zeros_like(ema13_1d_aligned, dtype=bool)
    ema13_falling = np.zeros_like(ema13_1d_aligned, dtype=bool)
    ema13_rising[1:] = ema13_1d_aligned[1:] > ema13_1d_aligned[:-1]
    ema13_falling[1:] = ema13_1d_aligned[1:] < ema13_1d_aligned[:-1]
    
    # Elder Ray components: Bull Power = High - EMA13(1d), Bear Power = EMA13(1d) - Low
    bull_power = high - ema13_1d_aligned
    bear_power = ema13_1d_aligned - low
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 1)  # Sufficient warmup for EMA13
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema13_1d_aligned[i]) or np.isnan(ema13_rising[i]) or 
            np.isnan(ema13_falling[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Bull Power > 0, 1d EMA13 rising, volume filter
            long_cond = (bull_power[i] > 0) and ema13_rising[i] and volume_filter[i]
            # Short conditions: Bear Power > 0, 1d EMA13 falling, volume filter
            short_cond = (bear_power[i] > 0) and ema13_falling[i] and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bull Power < 0 (bulls losing strength)
            if bull_power[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bear Power < 0 (bears losing strength)
            if bear_power[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals