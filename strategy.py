#!/usr/bin/env python3
"""
4h_KAMA_Trend_RSI_Chop
Hypothesis: On 4-hour timeframe, use KAMA (Kaufman Adaptive Moving Average) to detect trend direction, combined with RSI for momentum and Choppiness Index for regime filtering. Enter long when KAMA slope is up, RSI > 50, and chop indicates trending (CHOP < 38.2). Enter short when KAMA slope is down, RSI < 50, and chop indicates trending. Exit when conditions reverse. This avoids range-bound whipsaws and captures trends in both bull and bear markets. Low trade frequency expected (~20-40/year) due to multiple filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 4h data for KAMA calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Get 1d data for Choppiness Index calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) on 4h close
    close_4h = df_4h['close'].values
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close_4h, 10))  # |close[t] - close[t-10]|
    volatility = np.sum(np.abs(np.diff(close_4h)), axis=0)  # sum of |close[t] - close[t-1]| over 10 periods
    # Fix dimensions
    change = np.concatenate([np.full(10, np.nan), change])
    volatility = np.concatenate([np.full(10, np.nan), volatility])
    # Avoid division by zero
    er = np.divide(change, volatility, out=np.full_like(change, np.nan), where=volatility!=0)
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Calculate KAMA
    kama = np.full_like(close_4h, np.nan)
    kama[0] = close_4h[0]
    for i in range(1, len(close_4h)):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close_4h[i] - kama[i-1])
    
    # Align KAMA to lower timeframe (assume 4h is higher than target timeframe)
    # Since we're using 4h data for 4h timeframe, no alignment needed if timeframe is 4h
    # But we'll keep the structure for flexibility
    kama_4h = kama
    
    # Calculate RSI (14) on close
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Choppiness Index (14) on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[np.nan], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[np.nan], close_1d[:-1]]))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    # ATR (14)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    # Sum of TR over 14 periods
    sum_tr = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    # Chop = 100 * log10(sum_tr / (hh - ll)) / log10(14)
    # Avoid division by zero and log of zero
    denominator = hh - ll
    chop = np.full_like(sum_tr, np.nan)
    mask = (denominator > 0) & (~np.isnan(sum_tr))
    chop[mask] = 100 * np.log10(sum_tr[mask] / denominator[mask]) / np.log10(14)
    
    # Align 1d chop to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # KAMA slope (direction) - compare current vs 3 periods ago
    kama_slope = kama_4h - np.concatenate([[np.nan, np.nan, np.nan], kama_4h[:-3]])
    
    # Conditions
    kama_up = kama_slope > 0
    kama_down = kama_slope < 0
    rsi_long = rsi > 50
    rsi_short = rsi < 50
    chop_trending = chop_aligned < 38.2  # Trending when chop < 38.2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_slope[i]) or np.isnan(rsi[i]) or 
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions
        long_entry = kama_up[i] and rsi_long[i] and chop_trending[i]
        short_entry = kama_down[i] and rsi_short[i] and chop_trending[i]
        
        # Exit when conditions reverse
        long_exit = kama_down[i] or rsi_short[i] or not chop_trending[i]
        short_exit = kama_up[i] or rsi_long[i] or not chop_trending[i]
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Close long
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.25   # Close short
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_KAMA_Trend_RSI_Chop"
timeframe = "4h"
leverage = 1.0