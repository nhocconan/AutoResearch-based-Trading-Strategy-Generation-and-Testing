#!/usr/bin/env python3
"""
1d_KAMA_Trend_With_Volume_and_Chop_Filter
Hypothesis: Use Kaufman Adaptive Moving Average (KAMA) to identify trend direction on daily timeframe.
Enter long when price > KAMA and volume is above average in non-choppy markets (Choppiness Index < 50).
Enter short when price < KAMA and volume is above average in non-choppy markets.
Exit when trend reverses or market becomes choppy (Choppiness Index >= 50).
Designed for 1d timeframe to target 10-25 trades/year with low frequency and high conviction.
Works in bull markets by riding trends and in bear markets by avoiding false signals via chop filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_kama(close, er_length=10, fast_sc=2, slow_sc=30):
    """Calculate Kaufman Adaptive Moving Average"""
    n = len(close)
    kama = np.zeros(n)
    if n == 0:
        return kama
    
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=er_length))
    volatility = np.sum(np.abs(np.diff(close)), axis=0) if er_length == 1 else \
                 np.array([np.sum(np.abs(np.diff(close[i:i+er_length]))) 
                          for i in range(n-er_length+1)])
    # Pad volatility array
    volatility = np.concatenate([np.full(er_length-1, np.nan), volatility])
    er = np.where(volatility != 0, change / volatility, 0)
    
    # Smoothing constants
    sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
    
    # Initialize KAMA
    kama[0] = close[0]
    for i in range(1, n):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_choppiness(high, low, close, period=14):
    """Calculate Choppiness Index"""
    n = len(high)
    chop = np.full(n, np.nan)
    
    if n < period:
        return chop
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Sum of TR over period
    tr_sum = np.zeros(n)
    for i in range(period-1, n):
        tr_sum[i] = np.sum(tr[i-period+1:i+1])
    
    # Highest high and lowest low over period
    maxh = np.zeros(n)
    minl = np.zeros(n)
    for i in range(period-1, n):
        maxh[i] = np.max(high[i-period+1:i+1])
        minl[i] = np.min(low[i-period+1:i+1])
    
    # Choppiness formula
    for i in range(period-1, n):
        if maxh[i] != minl[i]:
            chop[i] = 100 * np.log10(tr_sum[i] / (maxh[i] - minl[i])) / np.log10(period)
        else:
            chop[i] = 50  # neutral if no range
    
    return chop

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data once for KAMA, chop, and weekly trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Weekly trend filter (1w timeframe)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    # Simple 10-period EMA for weekly trend
    ema10_1w = np.zeros_like(close_1w)
    if len(close_1w) >= 10:
        ema10_1w[9] = np.mean(close_1w[:10])
        multiplier = 2 / (10 + 1)
        for i in range(10, len(close_1w)):
            ema10_1w[i] = (close_1w[i] - ema10_1w[i-1]) * multiplier + ema10_1w[i-1]
    ema10_1w_aligned = align_htf_to_ltf(prices, df_1w, ema10_1w)
    
    # Daily indicators
    kama = calculate_kama(close_1d, er_length=10, fast_sc=2, slow_sc=30)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    chop = calculate_choppiness(high_1d, low_1d, close_1d, period=14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(ema10_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume filter: current volume > 1.3 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 1.3 * vol_ma
        else:
            volume_ok = False
        
        # Chop filter: avoid choppy markets (Choppiness Index >= 50)
        chop_ok = chop_aligned[i] < 50
        
        if position == 0:
            # Long conditions: price > KAMA, volume ok, not choppy, weekly uptrend
            if (price > kama_aligned[i] and volume_ok and chop_ok and 
                price > ema10_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price < KAMA, volume ok, not choppy, weekly downtrend
            elif (price < kama_aligned[i] and volume_ok and chop_ok and 
                  price < ema10_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: trend reversal or choppy market
            if price < kama_aligned[i] or not chop_ok:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend reversal or choppy market
            if price > kama_aligned[i] or not chop_ok:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_Trend_With_Volume_and_Chop_Filter"
timeframe = "1d"
leverage = 1.0