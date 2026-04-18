#!/usr/bin/env python3
"""
12h KAMA + 1d ATR volatility filter + volume confirmation
KAMA adapts to market noise - trending when efficiency ratio high, mean-reverting when low.
In trending markets: follow KAMA direction with volume confirmation.
In ranging markets: fade extreme deviations from KAMA with volume confirmation.
ATR filter ensures we only trade when volatility is sufficient.
Designed for 15-30 trades/year on 12h timeframe to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_kama(close, er_len=10, fast=2, slow=30):
    """Calculate Kaufman Adaptive Moving Average."""
    n = len(close)
    kama = np.full(n, np.nan)
    if n < 1:
        return kama
    
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=er_len))
    abs_change = np.sum(np.abs(np.diff(close)), axis=0) if hasattr(np.diff(close), 'shape') else None
    if abs_change is None:
        # Manual calculation for ER
        er = np.zeros(n)
        for i in range(er_len, n):
            if i >= er_len:
                price_change = abs(close[i] - close[i-er_len])
                total_change = 0
                for j in range(1, er_len+1):
                    total_change += abs(close[i-j+1] - close[i-j])
                if total_change != 0:
                    er[i] = price_change / total_change
                else:
                    er[i] = 0
    else:
        # Vectorized if possible
        er = np.zeros(n)
        for i in range(er_len, n):
            price_change = abs(close[i] - close[i-er_len])
            total_change = np.sum(np.abs(np.diff(close[i-er_len:i+1])))
            if total_change != 0:
                er[i] = price_change / total_change
            else:
                er[i] = 0
    
    # Smoothing constants
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
    
    # Initialize KAMA
    kama[0] = close[0]
    for i in range(1, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    return kama

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
    n = len(high)
    tr = np.zeros(n)
    for i in range(n):
        if i == 0:
            tr[i] = high[i] - low[i]
        else:
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = np.zeros(n)
    for i in range(n):
        if i < period:
            atr[i] = np.nan
        elif i == period:
            atr[i] = np.mean(tr[:period+1])
        else:
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ATR and KAMA context
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ATR(14) for volatility filter
    atr_14_1d = calculate_atr(high_1d, low_1d, close_1d, 14)
    
    # Calculate 1d KAMA for trend context
    kama_1d = calculate_kama(close_1d, er_len=10, fast=2, slow=30)
    
    # Align to 12h timeframe
    atr_14_1d_12h = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    kama_1d_12h = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # Calculate 12h KAMA for entry signals
    kama_12h = calculate_kama(close, er_len=10, fast=2, slow=30)
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 30)  # need sufficient data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(kama_12h[i]) or np.isnan(kama_1d_12h[i]) or 
            np.isnan(atr_14_1d_12h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # Volatility filter: ATR > 0.5 * 20-period average ATR
        if i >= 20:
            atr_ma = np.mean(atr_14_1d_12h[i-20:i]) if not np.isnan(atr_14_1d_12h[i-20:i]).any() else 0
            vol_filter = atr_14_1d_12h[i] > 0.5 * atr_ma if atr_ma > 0 else False
        else:
            vol_filter = True  # Not enough data for MA, allow trade
        
        if position == 0:
            # Determine market regime: trending if price away from KAMA, ranging if near
            kama_dist = abs(close[i] - kama_12h[i]) / kama_12h[i] if kama_12h[i] != 0 else 0
            
            if kama_dist > 0.015:  # Trending regime (>1.5% deviation)
                # Follow KAMA direction
                if close[i] > kama_12h[i] and vol_confirmed and vol_filter:
                    signals[i] = 0.25
                    position = 1
                elif close[i] < kama_12h[i] and vol_confirmed and vol_filter:
                    signals[i] = -0.25
                    position = -1
            else:  # Ranging regime (<1.5% deviation)
                # Fade extreme deviations from 1d KAMA
                if close[i] > kama_1d_12h[i] * 1.02 and vol_confirmed and vol_filter:  # 2% above
                    signals[i] = -0.20  # Short
                    position = -1
                elif close[i] < kama_1d_12h[i] * 0.98 and vol_confirmed and vol_filter:  # 2% below
                    signals[i] = 0.20   # Long
                    position = 1
        
        elif position == 1:
            # Long exit: price crosses below 12h KAMA or volatility drops
            if close[i] <= kama_12h[i] or (atr_14_1d_12h[i] < 0.3 * atr_ma if i >= 20 and 'atr_ma' in locals() else False):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above 12h KAMA or volatility drops
            if close[i] >= kama_12h[i] or (atr_14_1d_12h[i] < 0.3 * atr_ma if i >= 20 and 'atr_ma' in locals() else False):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_KAMA_ATR_Volume"
timeframe = "12h"
leverage = 1.0