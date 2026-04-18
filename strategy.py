#!/usr/bin/env python3
"""
12h_KAMA_Trend_RangeFilter
Hypothesis: Kaufman Adaptive Moving Average (KAMA) adapts to market noise, providing a dynamic trend filter. 
Combined with a range filter (ADX < 25) to avoid choppy markets and volume confirmation, this strategy aims to capture 
trending moves in both bull and bear markets while avoiding false signals in ranging conditions. 
Target: 15-30 trades/year on 12h timeframe with disciplined entry conditions.
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
    volume = prices['volume'].values
    
    # Calculate KAMA (10, 2, 30) on close
    def kama(close, length=10, fast=2, slow=30):
        direction = np.abs(np.diff(close, n=length))
        volatility = np.sum(np.abs(np.diff(close)), axis=1)
        er = np.where(volatility != 0, direction / volatility, 0)
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama = np.full_like(close, np.nan)
        kama[length] = close[length]
        for i in range(length+1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    kama_vals = kama(close, 10, 2, 30)
    
    # Calculate ADX (14) for range filter
    def adx(high, low, close, length=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(high[i] - high[i-1], 0)
            minus_dm[i] = max(low[i-1] - low[i], 0)
            if plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            elif minus_dm[i] < plus_dm[i]:
                minus_dm[i] = 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        atr = np.zeros_like(high)
        atr[length] = np.mean(tr[1:length+1])
        for i in range(length+1, len(tr)):
            atr[i] = (atr[i-1] * (length-1) + tr[i]) / length
        
        plus_di = 100 * (np.convolve(plus_dm, np.ones(length)/length, mode='same') / atr)
        minus_di = 100 * (np.convolve(minus_dm, np.ones(length)/length, mode='same') / atr)
        dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
        adx = np.convolve(dx, np.ones(length)/length, mode='same')
        return adx
    
    adx_vals = adx(high, low, close, 14)
    
    # Volume spike: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        if (np.isnan(kama_vals[i]) or np.isnan(adx_vals[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above KAMA, ADX < 25 (not trending strongly), volume spike
            if (close[i] > kama_vals[i] and adx_vals[i] < 25 and vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA, ADX < 25 (not trending strongly), volume spike
            elif (close[i] < kama_vals[i] and adx_vals[i] < 25 and vol_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below KAMA or ADX rises above 25 (trend weakening)
            if (close[i] < kama_vals[i] or adx_vals[i] > 25):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above KAMA or ADX rises above 25 (trend weakening)
            if (close[i] > kama_vals[i] or adx_vals[i] > 25):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_KAMA_Trend_RangeFilter"
timeframe = "12h"
leverage = 1.0