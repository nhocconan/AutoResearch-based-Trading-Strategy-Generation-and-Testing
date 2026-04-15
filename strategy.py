#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h KAMA + RSI + Chop regime filter
# Uses KAMA (Kaufman Adaptive Moving Average) for trend direction, RSI for momentum,
# and Choppiness Index to filter ranging markets. Trades only in trending regimes (CHOP < 38.2)
# with KAMA direction and RSI > 50 for longs, RSI < 50 for shorts.
# Designed for 12h timeframe to target 50-150 total trades over 4 years.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data for Chop filter and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate KAMA (10-period ER, 2 and 30 for SC) on 12h
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    
    # Efficiency Ratio
    change = np.abs(close_12h - np.roll(close_12h, 10))
    change[0:10] = 0  # First 10 values
    volatility = np.sum(np.abs(np.diff(close_12h, prepend=close_12h[0])), axis=0)
    # Calculate volatility using rolling sum of absolute changes
    volatility_arr = np.zeros_like(close_12h)
    for i in range(1, len(close_12h)):
        volatility_arr[i] = volatility_arr[i-1] + np.abs(close_12h[i] - close_12h[i-1])
        if i >= 10:
            volatility_arr[i] -= np.abs(close_12h[i-10] - close_12h[i-11]) if i >= 11 else 0
    volatility = volatility_arr
    er = np.where(volatility != 0, change / volatility, 0)
    
    # Smoothing Constants
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2
    
    # KAMA
    kama = np.full_like(close_12h, np.nan)
    kama[9] = close_12h[9]  # Start after 10 periods
    for i in range(10, len(close_12h)):
        if np.isnan(kama[i-1]):
            kama[i] = close_12h[i]
        else:
            kama[i] = kama[i-1] + sc[i] * (close_12h[i] - kama[i-1])
    
    # Calculate RSI (14) on 1d
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Choppiness Index (14) on 1d
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # ATR (14)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of TR over 14 periods
    sum_tr = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # MaxHigh - MinLow over 14 periods
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    range_maxmin = max_high - min_low
    
    # Chop = 100 * log10(sum_tr / range_maxmin) / log10(14)
    chop = 100 * np.log10(sum_tr / (range_maxmin + 1e-10)) / np.log10(14)
    
    # Align indicators to 12h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_12h, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or
            np.isnan(chop_aligned[i]) or np.isnan(close_12h[i//1]) if hasattr(df_12h, 'index') else False):
            continue
        
        # Only trade in trending regime (Chop < 38.2)
        if chop_aligned[i] >= 38.2:
            # Exit if in choppy market
            if position != 0:
                position = 0
                signals[i] = 0.0
            continue
        
        # Long entry: price above KAMA and RSI > 50
        if (close[i] > kama_aligned[i] and
            rsi_aligned[i] > 50 and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price below KAMA and RSI < 50
        elif (close[i] < kama_aligned[i] and
              rsi_aligned[i] < 50 and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: opposite signal
        elif position == 1 and (close[i] < kama_aligned[i] or rsi_aligned[i] < 50):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > kama_aligned[i] or rsi_aligned[i] > 50):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "12h_KAMA_RSI_Chop"
timeframe = "12h"
leverage = 1.0