#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Elder Ray Bull/Bear Power with Trend Filter.
# Uses 13-period EMA as trend filter and Elder Ray indicators (Bull Power = High - EMA13, Bear Power = EMA13 - Low).
# Long when Bull Power > 0 and rising + price above EMA13 (uptrend).
# Short when Bear Power > 0 and rising + price below EMA13 (downtrend).
# Volume filter (current volume > 1.3x 20-period average) ensures quality signals.
# Works in both bull and bear markets by following the trend via EMA13.
# Target: 75-200 trades over 4 years (19-50/year).

name = "6h_elder_ray_trend_filter_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # EMA13 for trend
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low
    bull_power = high - ema13
    bear_power = ema13 - low
    
    # Slope of Bull/Bear Power (3-period change)
    bull_slope = np.full(n, np.nan)
    bear_slope = np.full(n, np.nan)
    for i in range(3, n):
        bull_slope[i] = bull_power[i] - bull_power[i-3]
        bear_slope[i] = bear_power[i] - bear_power[i-3]
    
    # Volume filter: current volume > 1.3x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(bull_slope[i]) or np.isnan(bear_slope[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.3
        
        # Check exits
        if position == 1:  # long position
            # Exit: trend breaks or Bear Power dominates
            if (close[i] <= ema13[i] or 
                bear_power[i] > bull_power[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: trend breaks or Bull Power dominates
            if (close[i] >= ema13[i] or 
                bull_power[i] > bear_power[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation
            if volume_filter:
                # Long: Bull Power positive and rising + uptrend
                if (bull_power[i] > 0 and 
                    bull_slope[i] > 0 and 
                    close[i] > ema13[i]):
                    signals[i] = 0.25
                    position = 1
                # Short: Bear Power positive and rising + downtrend
                elif (bear_power[i] > 0 and 
                      bear_slope[i] > 0 and 
                      close[i] < ema13[i]):
                    signals[i] = -0.25
                    position = -1
    
    return signals