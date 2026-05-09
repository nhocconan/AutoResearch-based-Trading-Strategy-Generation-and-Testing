#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h KAMA trend following with 12h volatility regime filter.
# Uses Kaufman Adaptive Moving Average (KAMA) to reduce whipsaw in sideways markets.
# 12h ATR-based volatility filter ensures trades only occur in trending regimes.
# Designed for low trade frequency (<30/year) to minimize fee drag while capturing major trends.
# Works in both bull and bear markets by following the trend direction.
name = "4h_KAMA_Trend_VolatilityFilter_12h"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 12h data for volatility regime filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate KAMA on 4h close
    # Efficiency ratio: |close - close[10]| / sum(|close[i] - close[i-1]| for i in 1..10)
    change = np.abs(close - np.roll(close, 10))
    change[0:10] = np.nan  # Not enough data for first 10 periods
    
    volatility = np.zeros(n)
    for i in range(10, n):
        volatility[i] = np.nansum(np.abs(close[i-9:i+1] - np.roll(close[i-9:i+1], 1)))
    
    # Avoid division by zero
    er = np.divide(change, volatility, out=np.full_like(change, np.nan), where=volatility!=0)
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.full(n, np.nan)
    kama[0] = close[0]
    for i in range(1, n):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Align KAMA to ensure no look-ahead (already calculated on past data)
    kama_aligned = kama  # KAMA uses only past data, no alignment needed
    
    # 12h ATR for volatility regime filter
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr1[0] = np.nan  # First period has no previous close
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14)
    atr_12h = np.full(len(tr), np.nan)
    for i in range(14, len(tr)):
        if i == 14:
            atr_12h[i] = np.nanmean(tr[1:i+1])
        else:
            atr_12h[i] = (atr_12h[i-1] * 13 + tr[i]) / 14
    
    # Align ATR to 4h timeframe
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    
    # Volatility regime: ATR > 20-period EMA of ATR (trending market)
    atr_ema20 = np.full(len(atr_12h_aligned), np.nan)
    for i in range(len(atr_12h_aligned)):
        if i < 20:
            atr_ema20[i] = np.nan
        else:
            atr_ema20[i] = np.nanmean(atr_12h_aligned[i-19:i+1])
    
    volatile = atr_12h_aligned > atr_ema20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(kama_aligned[i]) or np.isnan(atr_12h_aligned[i]) or 
            np.isnan(atr_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: price crosses above KAMA in volatile (trending) market
            if price > kama_aligned[i] and volatile[i]:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below KAMA in volatile (trending) market
            elif price < kama_aligned[i] and volatile[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below KAMA
            if price < kama_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above KAMA
            if price > kama_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals