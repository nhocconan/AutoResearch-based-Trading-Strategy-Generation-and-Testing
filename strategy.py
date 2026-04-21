#!/usr/bin/env python3
"""
6h_1d_Pivot_SR_Fade_Zone
Hypothesis: Price shows mean-reversion behavior near daily pivot support/resistance zones (R1/S1, R2/S2) on 6b timeframe. Fades occur when price reaches these levels with momentum exhaustion (RSI divergence). Works in both bull/bear markets as pivots adapt to price levels, capturing reversals at key institutional levels.
"""

import numpy as np
import pandas as pd
from mtd_data import get_htf_data, align_htf_to_ltf

def calculate_rsi(close, period=14):
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    for i in range(len(close)):
        if i < period:
            avg_gain[i] = np.mean(gain[max(0, i-period+1):i+1]) if i > 0 else 0
            avg_loss[i] = np.mean(loss[max(0, i-period+1):i+1]) if i > 0 else 0
        else:
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_pivot_points(high, low, close):
    """Calculate daily pivot points and support/resistance levels"""
    pivot = (high + low + close) / 3
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    return pivot, r1, s1, r2, s2

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data once for pivot points and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily pivot points
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot_1d, r1_1d, s1_1d, r2_1d, s2_1d = calculate_pivot_points(high_1d, low_1d, close_1d)
    
    # Calculate RSI on daily data for momentum exhaustion
    rsi_1d = calculate_rsi(close_1d, 14)
    
    # Align all daily levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # 6s price data
    close_6h = prices['close'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if NaN in critical values
        if np.isnan(rsi_aligned[i]) or np.isnan(pivot_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_6h[i]
        rsi = rsi_aligned[i]
        
        # Define proximity to S/R levels (within 0.5% of level)
        proximity = 0.005
        
        near_s1 = abs(price - s1_aligned[i]) / s1_aligned[i] < proximity
        near_s2 = abs(price - s2_aligned[i]) / s2_aligned[i] < proximity
        near_r1 = abs(price - r1_aligned[i]) / r1_aligned[i] < proximity
        near_r2 = abs(price - r2_aligned[i]) / r2_aligned[i] < proximity
        
        # Momentum exhaustion conditions
        rsi_overbought = rsi > 65
        rsi_oversold = rsi < 35
        
        if position == 0:
            # Long setup: near support with RSI oversold
            if (near_s1 or near_s2) and rsi_oversold:
                signals[i] = 0.25
                position = 1
            # Short setup: near resistance with RSI overbought
            elif (near_r1 or near_r2) and rsi_overbought:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price reaches pivot or RSI becomes overbought
            if price >= pivot_aligned[i] or rsi > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price reaches pivot or RSI becomes oversold
            if price <= pivot_aligned[i] or rsi < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1d_Pivot_SR_Fade_Zone"
timeframe = "6h"
leverage = 1.0