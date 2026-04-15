#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA + RSI + Chop regime
# Uses KAMA for adaptive trend direction, RSI for momentum, and Choppiness Index for regime filter.
# Goes long when KAMA up, RSI > 50, and choppy market (CHOP > 61.8) for mean reversion.
# Goes short when KAMA down, RSI < 50, and choppy market (CHOP > 61.8).
# Works in sideways markets and avoids strong trends where mean reversion fails.
# Target: 30-100 total trades over 4 years.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load 1w data for Chop regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate KAMA (adaptive trend) on 1d
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Efficiency Ratio
    change = np.abs(np.diff(close_1d, n=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=0)  # placeholder - will compute properly below
    # Recompute volatility as sum of absolute changes over 10 periods
    volatility = np.zeros_like(close_1d)
    for i in range(10, len(close_1d)):
        volatility[i] = np.sum(np.abs(np.diff(close_1d[i-10:i+1])))
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2
    # KAMA calculation
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Calculate RSI (14) on 1d
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([np.full(14, np.nan), rsi])  # align with close_1d
    
    # Calculate Choppiness Index (14) on 1w
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first value
    
    # Sum of TR over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(sum(tr) / (hh - ll)) / log10(14)
    chop = 100 * np.log10(tr_sum / (hh - ll + 1e-10)) / np.log10(14)
    
    # Align indicators to lower timeframe (1d)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or
            np.isnan(chop_aligned[i])):
            continue
        
        # Long: KAMA up, RSI > 50, choppy market (CHOP > 61.8)
        if (kama_aligned[i] > kama_aligned[i-1] and
            rsi_aligned[i] > 50 and
            chop_aligned[i] > 61.8 and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short: KAMA down, RSI < 50, choppy market (CHOP > 61.8)
        elif (kama_aligned[i] < kama_aligned[i-1] and
              rsi_aligned[i] < 50 and
              chop_aligned[i] > 61.8 and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: opposite condition or chop < 38.2 (trending market)
        elif position == 1 and (kama_aligned[i] < kama_aligned[i-1] or chop_aligned[i] < 38.2):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (kama_aligned[i] > kama_aligned[i-1] or chop_aligned[i] < 38.2):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "1d_KAMA_RSI_Chop"
timeframe = "1d"
leverage = 1.0