#!/usr/bin/env python3
"""
1d_KAMA_Trend_With_Volume_And_Chop_Filter
Hypothesis: KAMA adapts to volatility and trend strength. Combined with volume confirmation and Choppiness Index filter,
it avoids whipsaws in choppy markets while capturing strong trends in both bull and bear regimes.
Daily timeframe reduces trade frequency to minimize fee drag. Target: 7-25 trades/year.
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
    
    # Get 1w data for trend filter and Choppiness Index
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # KAMA on daily close
    close_1w = df_1w['close'].values
    # Efficiency Ratio
    change = np.abs(np.diff(close_1w, 10))  # 10-period change
    # Sum of absolute differences
    abs_diff = np.abs(np.diff(close_1w, 1))
    volatility = pd.Series(abs_diff).rolling(window=10, min_periods=10).sum().values
    # Avoid division by zero
    er = np.zeros_like(close_1w)
    er[10:] = change[10:] / np.where(volatility[10:] == 0, 1, volatility[10:])
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2  # fast=2, slow=30
    kama = np.full_like(close_1w, np.nan)
    kama[9] = close_1w[9]  # start
    for i in range(10, len(close_1w)):
        kama[i] = kama[i-1] + sc[i] * (close_1w[i] - kama[i-1])
    
    # Align KAMA to daily timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1w, kama)
    
    # Choppiness Index on 1w
    # True Range
    tr1 = df_1w['high'] - df_1w['low']
    tr2 = np.abs(df_1w['high'] - df_1w['close'].shift(1))
    tr3 = np.abs(df_1w['low'] - df_1w['close'].shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # Sum of TR over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Highest high and lowest low over 14 periods
    hh = df_1w['high'].rolling(window=14, min_periods=14).max().values
    ll = df_1w['low'].rolling(window=14, min_periods=14).min().values
    # Chop calculation
    chop = np.full_like(close_1w, 50.0)  # default neutral
    mask = (hh - ll) > 0
    chop[14:] = 100 * np.log10(tr_sum[14:] / (hh[14:] - ll[14:])) / np.log10(14)
    
    # Align Chop to daily
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    
    # Volume confirmation: volume > 1.5 * 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for calculations
    start_idx = max(30, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        kama_val = kama_aligned[i]
        chop_val = chop_aligned[i]
        vol_conf = vol_confirm[i]
        
        # Only trade in trending markets (Chop < 61.8)
        if chop_val >= 61.8:
            # In choppy markets, stay flat
            signals[i] = 0.0
            position = 0
            continue
            
        if position == 0:
            # Long: price above KAMA and volume confirmation
            if close[i] > kama_val and vol_conf:
                signals[i] = size
                position = 1
            # Short: price below KAMA and volume confirmation
            elif close[i] < kama_val and vol_conf:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below KAMA
            if close[i] < kama_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above KAMA
            if close[i] > kama_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_KAMA_Trend_With_Volume_And_Chop_Filter"
timeframe = "1d"
leverage = 1.0