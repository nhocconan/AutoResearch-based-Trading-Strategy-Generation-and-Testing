#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) + 1d regime filter
# Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low
# In bull regime (price > 200 EMA): go long when Bull Power > 0 and rising
# In bear regime (price < 200 EMA): go short when Bear Power > 0 and rising
# Uses 13-period EMA for sensitivity, 200 EMA for regime filter
# Volume filter confirms institutional participation
# Target: 20-40 trades/year per symbol, works in bull/bear via regime adaptation

def generate_signals(prices):
    n = len(prices)
    if n < 250:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for regime filter (200 EMA) and Elder Ray calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # 1d EMA13 for Elder Ray
    close_1d = pd.Series(df_1d['close'].values)
    ema13_1d = close_1d.ewm(span=13, adjust=False, min_periods=13).mean().values
    ema13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema13_1d)
    
    # 1d EMA200 for regime filter
    ema200_1d = close_1d.ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Elder Ray components: Bull Power = High - EMA13, Bear Power = EMA13 - Low
    bull_power = high - ema13_1d_aligned
    bear_power = ema13_1d_aligned - low
    
    # Smooth Elder Ray with 3-period EMA to reduce noise
    bull_power_smooth = pd.Series(bull_power).ewm(span=3, adjust=False, min_periods=3).mean().values
    bear_power_smooth = pd.Series(bear_power).ewm(span=3, adjust=False, min_periods=3).mean().values
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 200
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema13_1d_aligned[i]) or np.isnan(ema200_1d_aligned[i]) or 
            np.isnan(bull_power_smooth[i]) or np.isnan(bear_power_smooth[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Regime determination
        bull_regime = close[i] > ema200_1d_aligned[i]
        bear_regime = close[i] < ema200_1d_aligned[i]
        
        # Elder Ray signals with momentum (current > previous)
        bull_momentum = bull_power_smooth[i] > bull_power_smooth[i-1]
        bear_momentum = bear_power_smooth[i] > bear_power_smooth[i-1]
        
        # Long conditions: bull regime + bull power positive + rising + volume
        if (bull_regime and 
            bull_power_smooth[i] > 0 and 
            bull_momentum and 
            volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short conditions: bear regime + bear power positive + rising + volume
        elif (bear_regime and 
              bear_power_smooth[i] > 0 and 
              bear_momentum and 
              volume_filter[i]):
            signals[i] = -0.25
            position = -1
        # Exit conditions: reverse signals or power deterioration
        elif position == 1 and (bull_power_smooth[i] <= 0 or not bull_momentum):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (bear_power_smooth[i] <= 0 or not bear_momentum):
            signals[i] = 0.0
            position = 0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

name = "6h_ElderRay_BullBearPower_1dRegime_Volume"
timeframe = "6h"
leverage = 1.0