#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Camarilla pivot levels from 1d: fade at R3/S3, breakout continuation at R4/S4
# Uses daily pivots for mean reversion at extremes and breakout confirmation
# Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag
name = "6h_camarilla_pivot_1d_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day
    # Based on previous day's high, low, close
    phigh = df_1d['high'].shift(1).values  # Previous day high
    plow = df_1d['low'].shift(1).values    # Previous day low
    pclose = df_1d['close'].shift(1).values # Previous day close
    
    # Pivot point and ranges
    pivot = (phigh + plow + pclose) / 3
    range_ = phigh - plow
    
    # Camarilla levels
    r3 = pivot + (range_ * 1.1 / 2)
    s3 = pivot - (range_ * 1.1 / 2)
    r4 = pivot + (range_ * 1.1)
    s4 = pivot - (range_ * 1.1)
    
    # Align to 6h timeframe (use previous day's levels)
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    r4_6h = align_htf_to_ltf(prices, df_1d, r4)
    s4_6h = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume confirmation: current 6h volume > 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or 
            np.isnan(r4_6h[i]) or np.isnan(s4_6h[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        vol_confirm = volume[i] > vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price reaches S3 (mean reversion target) OR breaks S4 (stop)
            if close[i] <= s3_6h[i] or close[i] <= s4_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price reaches R3 (mean reversion target) OR breaks R4 (stop)
            if close[i] >= r3_6h[i] or close[i] >= r4_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Fade at extremes: sell at R3, buy at S3 (mean reversion)
            if close[i] >= r3_6h[i] and vol_confirm:
                position = -1
                signals[i] = -0.25
            elif close[i] <= s3_6h[i] and vol_confirm:
                position = 1
                signals[i] = 0.25
            # Breakout continuation: buy above R4, sell below S4
            elif close[i] > r4_6h[i] and vol_confirm:
                position = 1
                signals[i] = 0.25
            elif close[i] < s4_6h[i] and vol_confirm:
                position = -1
                signals[i] = -0.25
    
    return signals