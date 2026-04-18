#!/usr/bin/env python3
"""
12h_KAMA_Trend_Follow_With_Volume_Confirmation
12h strategy using KAMA (Kaufman Adaptive Moving Average) for trend direction with volume confirmation.
- Long: Price above KAMA(14,2,30) + volume > 1.3x 20-period volume average
- Short: Price below KAMA(14,2,30) + volume > 1.3x 20-period volume average
- Exit: Opposite signal or volume drops below average
Designed for ~20-30 trades/year per symbol (80-120 total over 4 years)
Works in trending markets (both bull and bear) by following adaptive trend
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for KAMA calculation and volume average
    df_1d = get_htf_data(prices, '1d')
    
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) on daily data
    # KAMA parameters: ER period=10, Fast EMA=2, Slow EMA=30
    def calculate_kama(close_array, er_period=10, fast_ema=2, slow_ema=30):
        kama = np.full_like(close_array, np.nan, dtype=np.float64)
        if len(close_array) < er_period:
            return kama
        
        # Calculate Efficiency Ratio (ER)
        change = np.abs(np.diff(close_array, n=er_period))
        volatility = np.sum(np.abs(np.diff(close_array)), axis=0)
        er = np.where(volatility != 0, change / volatility, 0)
        
        # Calculate Smoothing Constant (SC)
        sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2
        
        # Calculate KAMA
        kama[er_period-1] = close_array[er_period-1]  # Initialize
        for i in range(er_period, len(close_array)):
            kama[i] = kama[i-1] + sc[i] * (close_array[i] - kama[i-1])
        
        return kama
    
    kama_1d = calculate_kama(close_1d)
    
    # Align KAMA to 12h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # Calculate 20-period volume average on daily data
    vol_ma_20 = np.full_like(volume_1d, np.nan, dtype=np.float64)
    for i in range(19, len(volume_1d)):
        vol_ma_20[i] = np.mean(volume_1d[i-19:i+1])
    
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # need enough for KAMA initialization
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(kama_aligned[i]) or np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend conditions
        price_above_kama = close[i] > kama_aligned[i]
        price_below_kama = close[i] < kama_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.3 * vol_ma_aligned[i]
        
        if position == 0:
            # Long: price above KAMA + volume confirmation
            if price_above_kama and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA + volume confirmation
            elif price_below_kama and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price below KAMA or volume drops below average
            if price_below_kama or not vol_confirm:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price above KAMA or volume drops below average
            if price_above_kama or not vol_confirm:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_KAMA_Trend_Follow_With_Volume_Confirmation"
timeframe = "12h"
leverage = 1.0