#!/usr/bin/env python3
"""
4h_KAMA_Adaptive_Trend_Strategy
Hypothesis: KAMA adapts to market noise, providing reliable trend signals in both trending and ranging markets. Combined with volume confirmation and ADX filter to avoid false signals during low volatility periods. Designed for low trade frequency (target: 20-50/year) with strong performance in both bull and bear markets.
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
    
    # Calculate KAMA (Kaufman Adaptive Moving Average)
    def kama(close, er_length=10, fast=2, slow=30):
        change = np.abs(np.diff(close, n=er_length))
        volatility = np.sum(np.abs(np.diff(close)), axis=1)
        er = np.zeros_like(close)
        er[er_length:] = change[er_length-1:] / np.maximum(volatility[er_length-1:], 1e-10)
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    kama_vals = kama(close)
    
    # Calculate ADX for trend strength
    def adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(high[i] - high[i-1], 0)
            minus_dm[i] = max(low[i-1] - low[i], 0)
            if plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            if minus_dm[i] < plus_dm[i]:
                minus_dm[i] = 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        atr = np.zeros_like(high)
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = 100 * np.zeros_like(high)
        minus_di = 100 * np.zeros_like(high)
        for i in range(period, len(high)):
            plus_di[i] = 100 * (np.sum(plus_dm[i-period+1:i+1]) / np.sum(tr[i-period+1:i+1]))
            minus_di[i] = 100 * (np.sum(minus_dm[i-period+1:i+1]) / np.sum(tr[i-period+1:i+1]))
        
        dx = np.zeros_like(high)
        for i in range(period, len(high)):
            dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / np.maximum(plus_di[i] + minus_di[i], 1e-10)
        
        adx_vals = np.zeros_like(high)
        adx_vals[2*period-1] = np.mean(dx[period:2*period])
        for i in range(2*period, len(high)):
            adx_vals[i] = (adx_vals[i-1] * (period-1) + dx[i]) / period
        
        return adx_vals
    
    adx_vals = adx(high, low, close)
    
    # Volume confirmation: volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)
    
    for i in range(start_idx, n):
        if np.isnan(kama[i]) or np.isnan(adx_vals[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above KAMA with strong trend (ADX > 25) and volume confirmation
            if close[i] > kama[i] and adx_vals[i] > 25 and vol_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA with strong trend (ADX > 25) and volume confirmation
            elif close[i] < kama[i] and adx_vals[i] > 25 and vol_confirm[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price below KAMA or trend weakens (ADX < 20)
            if close[i] < kama[i] or adx_vals[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price above KAMA or trend weakens (ADX < 20)
            if close[i] > kama[i] or adx_vals[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_KAMA_Adaptive_Trend_Strategy"
timeframe = "4h"
leverage = 1.0