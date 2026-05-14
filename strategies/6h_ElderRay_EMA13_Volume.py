#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Elder Ray Index (Bull/Bear Power) with 13-period EMA filter and volume confirmation.
# Bull Power = High - EMA13, Bear Power = EMA13 - Low
# Long when: Bull Power > 0 and rising, Bear Power < 0 and falling, volume > 1.5x 20-period average
# Short when: Bear Power > 0 and rising, Bull Power < 0 and falling, volume > 1.5x 20-period average
# Exit when: Bull Power crosses below 0 (for long) or Bear Power crosses below 0 (for short)
# Elder Ray measures bull/bear strength relative to trend, EMA13 filters direction, volume confirms strength.
# Works in bull (buy strength) and bear (sell weakness). Target: 15-25 trades/year per symbol.
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
    
    # Calculate EMA13 for trend
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = high - ema13  # Bull Power: High - EMA
    bear_power = ema13 - low   # Bear Power: EMA - Low
    
    # 20-period volume average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for volume MA calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema13[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        bull = bull_power[i]
        bear = bear_power[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        if position == 0:
            # Long entry: Bull Power > 0 and rising, Bear Power < 0 and falling, volume spike
            if (bull > 0 and bull > bull_power[i-1] and 
                bear < 0 and bear < bear_power[i-1] and vol > 1.5 * vol_ma):
                signals[i] = 0.25
                position = 1
            # Short entry: Bear Power > 0 and rising, Bull Power < 0 and falling, volume spike
            elif (bear > 0 and bear > bear_power[i-1] and 
                  bull < 0 and bull < bull_power[i-1] and vol > 1.5 * vol_ma):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Bull Power crosses below 0
            if bull < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Bear Power crosses below 0
            if bear < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals