#!/usr/bin/env python3
# Hypothesis: 6h ADX + Williams Alligator combination with price outside mouth
# Long when price > Alligator lips (green line) and ADX > 25 (trending)
# Short when price < Alligator jaws (red line) and ADX > 25 (trending)
# Exit when price re-enters Alligator mouth (between jaws and lips) or ADX < 20
# Uses Williams Alligator (SMMA of median price) for trend direction and ADX for trend strength
# Designed to capture strong trends while avoiding choppy markets
# Target: 50-150 total trades over 4 years (12-37/year) with size 0.25

name = "6h_ADX_Alligator_Trend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(data, period):
    """Smoothed Moving Average (SMMA)"""
    if len(data) < period:
        return np.full_like(data, np.nan, dtype=float)
    result = np.full_like(data, np.nan, dtype=float)
    # First value is SMA
    result[period-1] = np.mean(data[:period])
    # Subsequent values: SMMA = (prev_smma * (period-1) + current) / period
    for i in range(period, len(data)):
        result[i] = (result[i-1] * (period-1) + data[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate Williams Alligator (using median price)
    median_price = (high + low) / 2
    jaw = smma(median_price, 13)  # Blue line (slowest)
    teeth = smma(median_price, 8)  # Red line (middle)
    lips = smma(median_price, 5)   # Green line (fastest)
    
    # Calculate ADX
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    atr = smma(tr, 14)
    plus_di = smma(dm_plus, 14)
    minus_di = smma(dm_minus, 14)
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) != 0, 
                  np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100, 0)
    adx = smma(dx, 14)
    
    # Align indicators to 6h timeframe (they're already calculated on 6h data)
    # No need for HTF alignment since we're using same timeframe
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for ADX calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(adx[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price > lips (green) and ADX > 25 (strong trend)
            if close[i] > lips[i] and adx[i] > 25:
                signals[i] = 0.25
                position = 1
            # Enter short: price < jaw (blue) and ADX > 25 (strong trend)
            elif close[i] < jaw[i] and adx[i] > 25:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price re-enters mouth (below lips) or ADX < 20 (weak trend)
            if close[i] < lips[i] or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price re-enters mouth (above jaw) or ADX < 20 (weak trend)
            if close[i] > jaw[i] or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals