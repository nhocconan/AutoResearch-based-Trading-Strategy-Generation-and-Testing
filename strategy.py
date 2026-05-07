#!/usr/bin/env python3
# 6h_WilliamsAlligator_ElderRay_Signal
# Hypothesis: Combines Williams Alligator (trend) with Elder Ray (bull/bear power) on 6h timeframe.
# Alligator uses SMAs (13,8,5) with 8,5,3 period shifts to identify trend direction and strength.
# Elder Ray calculates Bull Power (high - EMA13) and Bear Power (low - EMA13) to measure buying/selling pressure.
# Long when: Bull Power > 0, price above Alligator Jaw (13-period SMMA shifted 8), and Bear Power rising.
# Short when: Bear Power < 0, price below Alligator Jaw, and Bull Power falling.
# Uses 1-week trend filter to avoid counter-trend trades in strong weekly trends.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

name = "6h_WilliamsAlligator_ElderRay_Signal"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(source, length):
    """Smoothed Moving Average (SMMA) - also called Wilder's Moving Average"""
    if length < 1:
        return source
    smma = np.full_like(source, np.nan, dtype=float)
    smma[length-1] = np.mean(source[:length])
    for i in range(length, len(source)):
        smma[i] = (smma[i-1] * (length-1) + source[i]) / length
    return smma

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Williams Alligator components (using SMMA)
    jaw = smma(close, 13)  # Blue line
    teeth = smma(close, 8)   # Red line
    lips = smma(close, 5)    # Green line
    
    # Shift the lines as per Alligator definition
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # Elder Ray components
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(jaw_shifted[i]) or np.isnan(teeth_shifted[i]) or 
            np.isnan(lips_shifted[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(ema_50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend from 1w EMA50
        uptrend_1w = close[i] > ema_50_1w_aligned[i]
        downtrend_1w = close[i] < ema_50_1w_aligned[i]
        
        if position == 0:
            # Long conditions: Bull Power > 0, price above Jaw, Bull Power rising (vs previous)
            if (bull_power[i] > 0 and 
                close[i] > jaw_shifted[i] and 
                i > 0 and bull_power[i] > bull_power[i-1] and
                uptrend_1w):  # Only long in 1w uptrend
                signals[i] = 0.25
                position = 1
            # Short conditions: Bear Power < 0, price below Jaw, Bear Power falling (vs previous)
            elif (bear_power[i] < 0 and 
                  close[i] < jaw_shifted[i] and 
                  i > 0 and bear_power[i] < bear_power[i-1] and
                  downtrend_1w):  # Only short in 1w downtrend
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bull Power <= 0 or price crosses below Teeth
            if bull_power[i] <= 0 or close[i] < teeth_shifted[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bear Power >= 0 or price crosses above Teeth
            if bear_power[i] >= 0 or close[i] > teeth_shifted[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals