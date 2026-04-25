#!/usr/bin/env python3
"""
1d KAMA + RSI + Chop Regime Filter
Hypothesis: Kaufman Adaptive Moving Average (KAMA) captures trend with low lag. 
Long when price > KAMA and RSI < 70 (avoid overbought), short when price < KAMA and RSI > 30 (avoid oversold).
Choppiness Index (CHOP) > 61.8 = range market (fade extremes), CHOP < 38.2 = trending (follow KAMA).
Uses 1d primary timeframe with 1w HTF for regime confirmation (weekly CHOP > 50 = range -> reduce size).
Target: 15-25 trades/year on 1d, discrete sizing 0.25, max drawdown controlled by regime filter.
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
    
    # Get 1d data for KAMA, RSI, CHOP (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate KAMA on 1d (ER=10, FAST=2, SLOW=30)
    close_1d = pd.Series(df_1d['close'])
    change = abs(close_1d.diff(10)).values
    volatility = close_1d.diff(1).abs().rolling(10, min_periods=1).sum().values
    er = np.where(volatility > 0, change / volatility, 0)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = np.full_like(close_1d, np.nan, dtype=float)
    kama[0] = close_1d.iloc[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d.iloc[i] - kama[i-1])
    kama_1d = kama
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # Calculate RSI(14) on 1d
    delta = close_1d.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    rsi_1d = rsi.values
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate Choppiness Index(14) on 1d
    atr = pd.Series(np.maximum(np.maximum(df_1d['high'] - df_1d['low'], 
                                          abs(df_1d['high'] - df_1d['close'].shift())),
                               abs(df_1d['low'] - df_1d['close'].shift())))
    atr_sum = atr.rolling(14, min_periods=14).sum()
    high_max = df_1d['high'].rolling(14, min_periods=14).max()
    low_min = df_1d['low'].rolling(14, min_periods=14).min()
    chop = 100 * np.log10(atr_sum / (high_max - low_min)) / np.log10(14)
    chop_1d = chop.values
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Get 1w data for weekly chop regime (additional delay for stability)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Weekly chop for regime filter
    atr_w = pd.Series(np.maximum(np.maximum(df_1w['high'] - df_1w['low'],
                                           abs(df_1w['high'] - df_1w['close'].shift())),
                                 abs(df_1w['low'] - df_1w['close'].shift())))
    atr_sum_w = atr_w.rolling(14, min_periods=14).sum()
    high_max_w = df_1w['high'].rolling(14, min_periods=14).max()
    low_min_w = df_1w['low'].rolling(14, min_periods=14).min()
    chop_w = 100 * np.log10(atr_sum_w / (high_max_w - low_min_w)) / np.log10(14)
    chop_w_1d = chop_w.values
    chop_w_1d_aligned = align_htf_to_ltf(prices, df_1w, chop_w_1d, additional_delay_bars=1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for all indicators
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_1d_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(chop_1d_aligned[i]) or np.isnan(chop_w_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        kama_val = kama_1d_aligned[i]
        rsi_val = rsi_1d_aligned[i]
        chop_val = chop_1d_aligned[i]
        chop_w_val = chop_w_1d_aligned[i]
        
        # Regime filter: weekly chop > 50 = range market -> reduce size or avoid
        is_range = chop_w_val > 50
        size = 0.15 if is_range else 0.25  # smaller size in ranging markets
        
        if position == 0:
            # Look for entry signals
            # Long: price > KAMA, RSI < 70 (not overbought), and not extreme chop
            long_entry = (curr_close > kama_val) and (rsi_val < 70) and (chop_val < 80)
            # Short: price < KAMA, RSI > 30 (not oversold), and not extreme chop
            short_entry = (curr_close < kama_val) and (rsi_val > 30) and (chop_val < 80)
            
            if long_entry:
                signals[i] = size
                position = 1
            elif short_entry:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price crosses below KAMA OR RSI > 75 (overbought exit)
            if curr_close < kama_val or rsi_val > 75:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short position management
            # Exit: price crosses above KAMA OR RSI < 25 (oversold exit)
            if curr_close > kama_val or rsi_val < 25:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_KAMA_RSI_ChopRegime"
timeframe = "1d"
leverage = 1.0