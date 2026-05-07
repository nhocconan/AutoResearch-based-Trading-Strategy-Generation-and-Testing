#!/usr/bin/env python3
# 4h_Chaikin_Money_Flow_Signal_v1
# Hypothesis: Uses Chaikin Money Flow (CMF) to measure buying/selling pressure with trend filter.
# Enters long when CMF > 0.15 (strong buying pressure) and price > 4h EMA50 (uptrend).
# Enters short when CMF < -0.15 (strong selling pressure) and price < 4h EMA50 (downtrend).
# Exits when CMF crosses back to zero or trend changes.
# CMF combines price and volume to confirm institutional activity, working in both bull and bear markets.
# Targets 20-35 trades/year (80-140 total over 4 years) with strict entry conditions.

name = "4h_Chaikin_Money_Flow_Signal_v1"
timeframe = "4h"
leverage = 1.0

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
    
    # Calculate Chaikin Money Flow (CMF) on 4h data
    # CMF = sum(Money Flow Volume * 20) / sum(Volume * 20)
    # Money Flow Multiplier = [(Close - Low) - (High - Close)] / (High - Low)
    # Money Flow Volume = Money Flow Multiplier * Volume
    
    # Avoid division by zero
    high_low = high - low
    high_low = np.where(high_low == 0, 1e-10, high_low)
    
    mf_multiplier = ((close - low) - (high - close)) / high_low
    mf_volume = mf_multiplier * volume
    
    # Calculate 20-period sums
    mf_volume_sum = pd.Series(mf_volume).rolling(window=20, min_periods=20).sum().values
    volume_sum = pd.Series(volume).rolling(window=20, min_periods=20).sum().values
    
    # CMF calculation
    cmf = np.where(volume_sum != 0, mf_volume_sum / volume_sum, 0.0)
    
    # Calculate 4h EMA50 for trend filter
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if np.isnan(cmf[i]) or np.isnan(ema_50[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Strong buying pressure (CMF > 0.15) and uptrend (price > EMA50)
            if cmf[i] > 0.15 and close[i] > ema_50[i]:
                signals[i] = 0.25
                position = 1
            # Short: Strong selling pressure (CMF < -0.15) and downtrend (price < EMA50)
            elif cmf[i] < -0.15 and close[i] < ema_50[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: CMF turns negative or trend turns down
            if cmf[i] < 0 or close[i] < ema_50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: CMF turns positive or trend turns up
            if cmf[i] > 0 or close[i] > ema_50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals