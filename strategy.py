#!/usr/bin/env python3
"""
6h_ADX_WilliamsAlligator_v1
Hypothesis: Combine ADX trend strength with Williams Alligator crossover on 6h timeframe. 
Go long when ADX > 25 (strong trend) + Alligator jaw < teeth < lips (bullish alignment).
Go short when ADX > 25 + Alligator jaw > teeth > lips (bearish alignment).
Exit when ADX < 20 (weakening trend) or Alligator lines crossover in opposite direction.
Position size: 0.25. Target: 50-150 total trades over 4 years = 12-37/year.
Uses Williams Alligator (SMAs of median price) which is proven in trending markets.
Works in bull (strong uptrend with bullish Alligator) and bear (strong downtrend with bearish Alligator).
"""

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
    
    # Calculate median price for Alligator
    median_price = (high + low) / 2.0
    
    # Williams Alligator lines (SMAs of median price)
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().shift(8)  # 13-period, shifted 8
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().shift(5)   # 8-period, shifted 5
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().shift(3)    # 5-period, shifted 3
    
    # ADX calculation
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        plus_dm[i] = high_diff if high_diff > low_diff and high_diff > 0 else 0
        minus_dm[i] = low_diff if low_diff > high_diff and low_diff > 0 else 0
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    def WilderSmoothing(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.nansum(data[:period])  # First value is simple average
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    period_adx = 14
    tr_smooth = WilderSmoothing(tr, period_adx)
    plus_dm_smooth = WilderSmoothing(plus_dm, period_adx)
    minus_dm_smooth = WilderSmoothing(minus_dm, period_adx)
    
    # Avoid division by zero
    plus_di = np.where(tr_smooth != 0, (plus_dm_smooth / tr_smooth) * 100, 0)
    minus_di = np.where(tr_smooth != 0, (minus_dm_smooth / tr_smooth) * 100, 0)
    
    dx = np.where((plus_di + minus_di) != 0, 
                  np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100, 
                  0)
    adx = WilderSmoothing(dx, period_adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Alligator (max 13+8=21) and ADX (14*3=42 for smoothing)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(adx[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Alligator alignment
        bullish_alignment = (jaw[i] < teeth[i]) and (teeth[i] < lips[i])
        bearish_alignment = (jaw[i] > teeth[i]) and (teeth[i] > lips[i])
        
        # ADX trend strength
        strong_trend = adx[i] > 25
        weak_trend = adx[i] < 20
        
        if position == 0:
            # Long setup: strong trend + bullish Alligator alignment
            if strong_trend and bullish_alignment:
                signals[i] = 0.25
                position = 1
            # Short setup: strong trend + bearish Alligator alignment
            elif strong_trend and bearish_alignment:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: weak trend OR bearish Alligator alignment
            if weak_trend or bearish_alignment:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: weak trend OR bullish Alligator alignment
            if weak_trend or bullish_alignment:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ADX_WilliamsAlligator_v1"
timeframe = "6h"
leverage = 1.0