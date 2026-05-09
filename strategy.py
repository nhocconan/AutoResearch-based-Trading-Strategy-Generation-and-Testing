#!/usr/bin/env python3
# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA trend filter and volume spike confirmation
# Long when Bull Power > 0, Bear Power < 0, EMA50 rising, volume > 1.5x average
# Short when Bull Power < 0, Bear Power > 0, EMA50 falling, volume > 1.5x average
# Exit when Bull Power and Bear Power both < 0 (weak momentum) or both > 0 (exhaustion)
# Elder Ray measures bull/bear strength relative to EMA, effective in trending and ranging markets
# Target: 50-150 total trades over 4 years (12-37/year) with size 0.25

name = "6h_ElderRay_EMA50_VolumeSpike"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA for Elder Ray calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 13-period EMA for Elder Ray (standard)
    ema13_1d = pd.Series(df_1d['close']).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Bull Power (High - EMA13) and Bear Power (Low - EMA13)
    bull_power = df_1d['high'].values - ema13_1d
    bear_power = df_1d['low'].values - ema13_1d
    
    # Align Elder Ray components to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: current volume > 1.5x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean()
    vol_confirm = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Need enough data for EMA calculations
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Bull Power > 0, Bear Power < 0, EMA50 rising, volume spike
            if (bull_power_aligned[i] > 0 and 
                bear_power_aligned[i] < 0 and 
                ema50_1d_aligned[i] > ema50_1d_aligned[i-1] and
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: Bull Power < 0, Bear Power > 0, EMA50 falling, volume spike
            elif (bull_power_aligned[i] < 0 and 
                  bear_power_aligned[i] > 0 and 
                  ema50_1d_aligned[i] < ema50_1d_aligned[i-1] and
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: both powers negative (weak momentum) or both positive (exhaustion)
            if (bull_power_aligned[i] < 0 and bear_power_aligned[i] < 0) or \
               (bull_power_aligned[i] > 0 and bear_power_aligned[i] > 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: both powers negative (weak momentum) or both positive (exhaustion)
            if (bull_power_aligned[i] < 0 and bear_power_aligned[i] < 0) or \
               (bull_power_aligned[i] > 0 and bear_power_aligned[i] > 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals