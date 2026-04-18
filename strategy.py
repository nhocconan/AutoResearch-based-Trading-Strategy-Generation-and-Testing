#!/usr/bin/env python3
"""
6h ADX + Williams Alligator Combination
Hypothesis: The Williams Alligator (three SMAs) identifies trend direction and strength,
while ADX filters for trending conditions. In trending markets (ADX > 25), we go long
when price is above the Alligator's jaws (13-period SMA) and short when below.
This combination works in both bull and bear markets by capturing sustained trends
while avoiding choppy periods. Target: 15-25 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Williams Alligator: three SMAs (Jaw=13, Teeth=8, Lips=5)
    close_series = pd.Series(close)
    jaw = close_series.rolling(window=13, min_periods=13).mean().values  # Blue line
    teeth = close_series.rolling(window=8, min_periods=8).mean().values   # Red line
    lips = close_series.rolling(window=5, min_periods=5).mean().values    # Green line
    
    # ADX for trend strength (14-period)
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    dm_plus14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    # Directional Indicators
    di_plus = np.where(tr14 > 0, 100 * dm_plus14 / tr14, 0)
    di_minus = np.where(tr14 > 0, 100 * dm_minus14 / tr14, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) > 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        adx_val = adx[i]
        
        # Alligator conditions: jaws (13) as main trend indicator
        # Teeth and lips help identify alignment but jaw is primary
        
        if position == 0:
            # Strong trend (ADX > 25) and price above Alligator's jaw = long
            if adx_val > 25 and price > jaw[i]:
                signals[i] = 0.25
                position = 1
            # Strong trend and price below Alligator's jaw = short
            elif adx_val > 25 and price < jaw[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long if trend weakens (ADX < 20) or price crosses below jaw
            if adx_val < 20 or price < jaw[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short if trend weakens (ADX < 20) or price crosses above jaw
            if adx_val < 20 or price > jaw[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ADX_WilliamsAlligator"
timeframe = "6h"
leverage = 1.0