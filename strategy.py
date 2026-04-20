#!/usr/bin/env python3
"""
4h_KAMA_RSI_With_Chop_Filter
Hypothesis: Use KAMA (Kaufman Adaptive Moving Average) to identify trend direction and RSI for momentum, filtered by Choppiness Index to avoid whipsaws in sideways markets. Long when KAMA turns up and RSI > 50 in trending markets; short when KAMA turns down and RSI < 50 in trending markets. Uses 12h timeframe for trend context to reduce noise and improve reliability. Designed to work in both bull and bear markets by avoiding trades in choppy conditions (Chop > 61.8) and only trading when clear trends exist (Chop < 38.2). Target: 20-40 trades/year with position size 0.25.
"""

name = "4h_KAMA_RSI_With_Chop_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) - trend identifier
    def kama(close, period=10, fast=2, slow=30):
        # Efficiency Ratio
        change = np.abs(np.diff(close, n=period))
        volatility = np.sum(np.abs(np.diff(close)), axis=0)
        er = np.zeros_like(close)
        er[period:] = change[period-1:] / volatility[period-1:]
        # Smoothing Constant
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        # KAMA
        kama_vals = np.zeros_like(close)
        kama_vals[:] = np.nan
        kama_vals[period] = close[period]
        for i in range(period+1, len(close)):
            kama_vals[i] = kama_vals[i-1] + sc[i] * (close[i] - kama_vals[i-1])
        return kama_vals
    
    # Calculate RSI (Relative Strength Index) - momentum
    def rsi(close, period=14):
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        for i in range(period+1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi_vals = 100 - (100 / (1 + rs))
        return rsi_vals
    
    # Calculate Choppiness Index - regime filter
    def choppiness_index(high, low, close, period=14):
        # True Range
        tr1 = np.abs(high - low)
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First period
        
        # Sum of True Range over period
        atr_sum = np.zeros_like(close)
        for i in range(period, len(close)):
            atr_sum[i] = np.sum(tr[i-period+1:i+1])
        
        # Highest high and lowest low over period
        hh = np.zeros_like(close)
        ll = np.zeros_like(close)
        for i in range(period-1, len(close)):
            hh[i] = np.max(high[i-period+1:i+1])
            ll[i] = np.min(low[i-period+1:i+1])
        
        # Choppiness Index
        chop = np.zeros_like(close)
        for i in range(period-1, len(close)):
            if hh[i] != ll[i]:
                chop[i] = 100 * np.log10(atr_sum[i] / (hh[i] - ll[i])) / np.log10(period)
            else:
                chop[i] = 50  # Avoid division by zero
        return chop
    
    # Get 12h data for trend context
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h KAMA for trend context
    kama_12h = kama(close_12h, period=10, fast=2, slow=30)
    kama_12h_slope = np.diff(kama_12h, prepend=kama_12h[0])
    kama_12h_slope = np.append(kama_12h_slope, kama_12h_slope[-1])  # Same length
    
    # Align 12h KAMA slope to 4h timeframe
    kama_12h_slope_aligned = align_htf_to_ltf(prices, df_12h, kama_12h_slope)
    
    # Calculate indicators on 4h data
    kama_4h = kama(close, period=10, fast=2, slow=30)
    rsi_4h = rsi(close, period=14)
    chop_4h = choppiness_index(high, low, close, period=14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_4h[i]) or np.isnan(rsi_4h[i]) or np.isnan(chop_4h[i]) or
            np.isnan(kama_12h_slope_aligned[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        # Only trade in trending markets (Chop < 38.2)
        if chop_4h[i] >= 38.2:
            # In choppy or trending markets, stay flat
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: KAMA turning up (positive slope) AND RSI > 50 AND 12h KAMA trending up
            if kama_4h[i] > kama_4h[i-1] and rsi_4h[i] > 50 and kama_12h_slope_aligned[i] > 0:
                signals[i] = 0.25
                position = 1
            # Short: KAMA turning down (negative slope) AND RSI < 50 AND 12h KAMA trending down
            elif kama_4h[i] < kama_4h[i-1] and rsi_4h[i] < 50 and kama_12h_slope_aligned[i] < 0:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: KAMA turning down OR RSI < 50
            if kama_4h[i] < kama_4h[i-1] or rsi_4h[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: KAMA turning up OR RSI > 50
            if kama_4h[i] > kama_4h[i-1] or rsi_4h[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals