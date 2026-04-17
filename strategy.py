#!/usr/bin/env python3
"""
1d_KAMA_RSI_ChopFilter_v1
Daily strategy combining Kaufman Adaptive Moving Average trend with RSI momentum and Choppiness Index regime filter.
Designed to work in both bull and bear markets by adapting to trending and ranging conditions.
Target: 30-100 total trades over 4 years (7-25/year).
"""

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
    
    # === Daily KAMA (10-period ER) ===
    close_s = pd.Series(close)
    change = abs(close_s.diff(1))
    volatility = change.rolling(window=10, min_periods=10).sum()
    er = change / volatility.replace(0, np.nan)
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        if not np.isnan(sc.iloc[i]):
            kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # === Daily RSI (14-period) ===
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # === Weekly Choppiness Index (14-period) ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    atr_1w = np.zeros(len(high_1w))
    tr_1w = np.maximum(
        high_1w[1:] - low_1w[1:],
        np.maximum(
            abs(high_1w[1:] - close_1w[:-1]),
            abs(low_1w[1:] - close_1w[:-1])
        )
    )
    atr_1w[1:] = tr_1w
    atr_1w[0] = high_1w[0] - low_1w[0]
    
    sum_atr_1w = pd.Series(atr_1w).rolling(window=14, min_periods=14).sum()
    highest_high_1w = pd.Series(high_1w).rolling(window=14, min_periods=14).max()
    lowest_low_1w = pd.Series(low_1w).rolling(window=14, min_periods=14).min()
    
    chop = 100 * np.log10(sum_atr_1w / (highest_high_1w - lowest_low_1w)) / np.log10(14)
    chop = chop.fillna(50).values
    
    # Align weekly chop to daily timeframe (wait for weekly close)
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price above KAMA, RSI > 50, and chop < 61.8 (trending)
            if (close[i] > kama[i] and 
                rsi[i] > 50 and 
                chop_aligned[i] < 61.8):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price below KAMA, RSI < 50, and chop < 61.8 (trending)
            elif (close[i] < kama[i] and 
                  rsi[i] < 50 and 
                  chop_aligned[i] < 61.8):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: reverse signal or chop > 61.8 (ranging)
        elif position == 1:
            # Exit long: price below KAMA or chop > 61.8
            if (close[i] < kama[i] or chop_aligned[i] > 61.8):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price above KAMA or chop > 61.8
            if (close[i] > kama[i] or chop_aligned[i] > 61.8):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_RSI_ChopFilter_v1"
timeframe = "1d"
leverage = 1.0