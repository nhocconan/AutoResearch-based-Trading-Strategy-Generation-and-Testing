#!/usr/bin/env python3
# Hypothesis: 4h timeframe with 12h KAMA trend + volume spike confirmation + RSI mean reversion.
# Uses 12h Kaufman Adaptive Moving Average (KAMA) for trend direction (efficient, low lag).
# Enters long when price > 12h KAMA + RSI < 30 + volume spike (>1.5x avg), short when price < 12h KAMA + RSI > 70 + volume spike.
# Exits when price crosses back below/above 12h KAMA or RSI reverts to neutral (40-60).
# Target: 80-150 total trades over 4 years (20-38/year) with size 0.25.

name = "4h_KAMA_Trend_RSI_MeanReversion_Volume"
timeframe = "4h"
leverage = 1.0

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
    volume = prices['volume'].values
    
    # Calculate 12h KAMA (trend direction)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # Calculate efficiency ratio
    change = np.abs(np.diff(close_12h, prepend=close_12h[0]))
    volatility = np.sum(np.abs(np.diff(close_12h)), axis=0)  # placeholder - will compute properly
    
    # Proper KAMA calculation
    def calculate_kama(close, length=10, fast=2, slow=30):
        # Direction
        direction = np.abs(np.diff(close, prepend=close[0]))
        # Volatility
        volatility = np.sum(np.abs(np.diff(close)), axis=0)  # This is incorrect, need rolling sum
        
        # Correct approach
        change = np.abs(np.diff(close, prepend=close[0]))
        # Volatility over 'length' period
        vol = np.zeros_like(close)
        for i in range(length, len(close)):
            vol[i] = np.sum(np.abs(np.diff(close[i-length:i+1])))
        
        # Avoid division by zero
        er = np.zeros_like(close)
        mask = vol != 0
        er[mask] = change[mask] / vol[mask]
        
        # Smoothing constants
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        
        # KAMA
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    # Simplified KAMA using exponential moving average of volatility
    # More practical implementation
    change_12h = np.abs(np.diff(close_12h, prepend=close_12h[0]))
    # Volatility over 10 periods
    vol_12h = np.zeros_like(close_12h)
    for i in range(10, len(close_12h)):
        vol_12h[i] = np.sum(np.abs(np.diff(close_12h[i-10:i+1])))
    
    # Efficiency ratio
    er_12h = np.zeros_like(close_12h)
    mask = vol_12h != 0
    er_12h[mask] = change_12h[mask] / vol_12h[mask]
    
    # Smoothing constants
    fast_sc = 2/(2+1)
    slow_sc = 2/(30+1)
    sc_12h = (er_12h * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA
    kama_12h = np.zeros_like(close_12h)
    kama_12h[0] = close_12h[0]
    for i in range(1, len(close_12h)):
        kama_12h[i] = kama_12h[i-1] + sc_12h[i] * (close_12h[i] - kama_12h[i-1])
    
    kama_12h_aligned = align_htf_to_ltf(prices, df_12h, kama_12h)
    
    # RSI (14) on 4h close
    def calculate_rsi(close, period=14):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        
        # Initial average
        if len(close) > period:
            avg_gain[period] = np.mean(gain[1:period+1])
            avg_loss[period] = np.mean(loss[1:period+1])
            
            # Wilder's smoothing
            for i in range(period+1, len(close)):
                avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
                avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        
        # Avoid division by zero
        rs = np.zeros_like(close)
        mask = avg_loss != 0
        rs[mask] = avg_gain[mask] / avg_loss[mask]
        
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi = calculate_rsi(close, 14)
    
    # Volume spike detection (>1.5x 20-period average)
    def calculate_volume_spike(volume, period=20):
        vol_ma = np.zeros_like(volume)
        for i in range(period, len(volume)):
            vol_ma[i] = np.mean(volume[i-period:i])
        
        spike = np.zeros_like(volume, dtype=bool)
        spike[period:] = volume[period:] > 1.5 * vol_ma[period:]
        return spike
    
    volume_spike = calculate_volume_spike(volume, 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama_12h_aligned[i]) or 
            np.isnan(rsi[i]) or 
            i < 20):  # volume spike needs 20 periods
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price > 12h KAMA + RSI < 30 + volume spike
            if close[i] > kama_12h_aligned[i] and rsi[i] < 30 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price < 12h KAMA + RSI > 70 + volume spike
            elif close[i] < kama_12h_aligned[i] and rsi[i] > 70 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below 12h KAMA OR RSI > 50 (mean reversion)
            if close[i] < kama_12h_aligned[i] or rsi[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above 12h KAMA OR RSI < 50 (mean reversion)
            if close[i] > kama_12h_aligned[i] or rsi[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals