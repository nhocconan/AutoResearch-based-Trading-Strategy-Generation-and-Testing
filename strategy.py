#!/usr/bin/env python3
"""
12h_WK1_Alligator_ElderRay_Squeeze
Hypothesis: Use 12h Williams Alligator for trend, Elder Ray for momentum, and Bollinger Band squeeze for breakout.
Long when price > Alligator teeth (Jaw), Bull Power > 0, and BB width breaks above 20-day percentile.
Short when price < Alligator teeth, Bear Power < 0, and BB width breaks above 20-day percentile.
Exit on trend reversal or momentum shift.
Designed for 12h timeframe with weekly/daily filters to limit trades to ~15-25/year.
Works in bull markets by buying strength and in bear markets by selling weakness.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_sma(arr, period):
    """Simple Moving Average"""
    sma = np.full_like(arr, np.nan)
    if len(arr) >= period:
        for i in range(period-1, len(arr)):
            sma[i] = np.mean(arr[i-period+1:i+1])
    return sma

def calculate_ewma(arr, period):
    """Exponential Weighted Moving Average"""
    ema = np.full_like(arr, np.nan)
    if len(arr) >= period:
        multiplier = 2 / (period + 1)
        ema[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            ema[i] = (arr[i] - ema[i-1]) * multiplier + ema[i-1]
    return ema

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for Alligator and Elder Ray
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Williams Alligator (13,8,5 smoothed with 8,5,3)
    jaw = calculate_sma(close_1w, 13)  # Blue line
    jaw = calculate_ewma(jaw, 8)       # Smoothed with 8-period
    teeth = calculate_sma(close_1w, 8) # Red line
    teeth = calculate_ewma(teeth, 5)   # Smoothed with 5-period
    lips = calculate_sma(close_1w, 5)  # Green line
    lips = calculate_ewma(lips, 3)     # Smoothed with 3-period
    
    jaw_aligned = align_htf_to_ltf(prices, df_1w, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1w, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1w, lips)
    
    # Elder Ray Power (13-period EMA)
    ema13 = calculate_ewma(close_1w, 13)
    bull_power = high_1w - ema13
    bear_power = low_1w - ema13
    bull_power_aligned = align_htf_to_ltf(prices, df_1w, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1w, bear_power)
    
    # Load daily data for Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Bollinger Bands (20,2)
    sma20 = calculate_sma(close_1d, 20)
    std20 = np.full_like(close_1d, np.nan)
    for i in range(19, len(close_1d)):
        std20[i] = np.std(close_1d[i-19:i+1])
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    bb_width = (upper - lower) / sma20
    
    # Bollinger Band squeeze: width > 20-day percentile (80th)
    bb_width_percentile = np.full_like(bb_width, np.nan)
    for i in range(19, len(bb_width)):
        window = bb_width[max(0, i-19):i+1]
        if len(window) >= 10:
            bb_width_percentile[i] = np.percentile(window, 80)
    bb_squeeze = bb_width > bb_width_percentile
    bb_squeeze_aligned = align_htf_to_ltf(prices, df_1d, bb_squeeze.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(bb_squeeze_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        
        # Trend: price relative to Alligator teeth
        price_vs_teeth = price > teeth_aligned[i]
        
        # Momentum: Elder Ray power
        bullish_momentum = bull_power_aligned[i] > 0
        bearish_momentum = bear_power_aligned[i] < 0
        
        # Breakout: Bollinger Band squeeze break
        breakout = bb_squeeze_aligned[i] > 0.5
        
        if position == 0:
            # Long conditions: uptrend + bullish momentum + breakout
            if price_vs_teeth and bullish_momentum and breakout:
                signals[i] = 0.25
                position = 1
            # Short conditions: downtrend + bearish momentum + breakout
            elif not price_vs_teeth and bearish_momentum and breakout:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: trend reversal or bearish momentum
            if not price_vs_teeth or not bullish_momentum:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend reversal or bullish momentum
            if price_vs_teeth or not bearish_momentum:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WK1_Alligator_ElderRay_Squeeze"
timeframe = "12h"
leverage = 1.0