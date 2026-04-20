#!/usr/bin/env python3
# 4h_1d_Alligator_ElderRay_Ribbon_Trend
# Hypothesis: Combine Williams Alligator (trend direction) with Elder Ray (bull/bear power) on 1d timeframe.
# Trade in direction of 1d trend using 4h timeframe for entry timing.
# Alligator: Jaw (13-bar SMMA), Teeth (8-bar SMMA), Lips (5-bar SMMA). 
# Bull Power = High - EMA13, Bear Power = EMA13 - Low.
# Enter long when: Lips > Teeth > Jaw (bullish alignment) AND Bull Power > 0 AND price > Lips.
# Enter short when: Jaw > Teeth > Lips (bearish alignment) AND Bear Power > 0 AND price < Teeth.
# Uses 1d trend filter to avoid counter-trend trades, works in bull/bear markets.
# Target: 20-50 trades/year (80-200 total over 4 years).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Alligator_ElderRay_Ribbon_Trend"
timeframe = "4h"
leverage = 1.0

def smma(data, period):
    """Smoothed Moving Average (SMMA) - same as Wilder's smoothing"""
    if len(data) < period:
        return np.full_like(data, np.nan, dtype=float)
    result = np.full_like(data, np.nan, dtype=float)
    # First value is simple average
    result[period-1] = np.mean(data[:period])
    # Subsequent values: (prev*(period-1) + current) / period
    for i in range(period, len(data)):
        result[i] = (result[i-1] * (period-1) + data[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop for Alligator and Elder Ray
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need enough for SMMA calculations
        return np.zeros(n)
    
    # === Calculate 1d Alligator (SMMA) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Alligator lines: Jaw (13), Teeth (8), Lips (5) - all SMMA of median price
    median_price_1d = (high_1d + low_1d) / 2.0
    jaw_1d = smma(median_price_1d, 13)  # Jaw (blue)
    teeth_1d = smma(median_price_1d, 8)  # Teeth (red)
    lips_1d = smma(median_price_1d, 5)   # Lips (green)
    
    # === Calculate 1d Elder Ray ===
    # EMA13 for Elder Ray
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power_1d = high_1d - ema13_1d  # Bull Power = High - EMA13
    bear_power_1d = ema13_1d - low_1d   # Bear Power = EMA13 - Low
    
    # === 4h: Price for entry timing ===
    close_4h = prices['close'].values
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    
    # Align all 1d indicators to 4h
    jaw_1d_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_1d_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_1d_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after warmup
        # Get values
        close_val = close_4h[i]
        high_val = high_4h[i]
        low_val = low_4h[i]
        jaw_val = jaw_1d_aligned[i]
        teeth_val = teeth_1d_aligned[i]
        lips_val = lips_1d_aligned[i]
        bull_power_val = bull_power_1d_aligned[i]
        bear_power_val = bear_power_1d_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(jaw_val) or np.isnan(teeth_val) or np.isnan(lips_val) or 
            np.isnan(bull_power_val) or np.isnan(bear_power_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bullish Alligator alignment + Bull Power positive + price above Lips
            if (lips_val > teeth_val and teeth_val > jaw_val and  # Lips > Teeth > Jaw
                bull_power_val > 0 and                            # Bull Power positive
                close_val > lips_val):                            # Price above Lips (entry confirmation)
                signals[i] = 0.25
                position = 1
            # Short: Bearish Alligator alignment + Bear Power positive + price below Teeth
            elif (jaw_val > teeth_val and teeth_val > lips_val and  # Jaw > Teeth > Lips
                  bear_power_val > 0 and                            # Bear Power positive
                  close_val < teeth_val):                           # Price below Teeth (entry confirmation)
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Bearish crossover (Teeth crosses below Lips) OR Bear Power turns negative
            if (teeth_val < lips_val or bear_power_val <= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Bullish crossover (Lips crosses above Teeth) OR Bull Power turns negative
            if (lips_val > teeth_val or bull_power_val <= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals