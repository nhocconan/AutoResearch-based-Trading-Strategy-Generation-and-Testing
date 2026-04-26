#!/usr/bin/env python3
"""
1d_KAMA_Direction_RSI_ChopFilter_v1
Hypothesis: On daily timeframe, KAMA trend direction + RSI(14) extreme + Choppiness index regime filter captures sustained moves while avoiding whipsaws in range/chop markets. Works in bull/bear via KAMA's adaptive smoothing. Designed for 1d to target 7-25 trades/year with discrete sizing (0.25).
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
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for regime filter (chop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # KAMA on 1d close (trend direction)
    close_1d = df_1d['close'].values
    # Efficiency Ratio
    change = np.abs(np.diff(close_1d, n=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close_1d, n=1)), axis=1)  # 10-period volatility
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    # KAMA calculation
    kama = np.full_like(close_1d, np.nan, dtype=float)
    kama[29] = close_1d[29]  # seed
    for i in range(30, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    kama_1d = kama
    
    # Align KAMA to 1d timeframe (already 1d, but using helper for consistency)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1w, kama_1d)  # using 1w as HTF for alignment logic
    
    # RSI(14) on 1d close
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([np.full(14, np.nan), rsi])  # align length
    
    # Align RSI to 1d timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1w, rsi)
    
    # Choppiness Index (14) on 1w data
    atr_1w = []
    tr_1w = np.maximum(df_1w['high'].values, np.roll(df_1w['close'].values, 1))
    tr_1w = np.maximum(tr_1w, df_1w['low'].values)
    tr_1w = np.maximum(tr_1w, np.roll(df_1w['close'].values, 1)) - np.minimum(df_1w['low'].values, np.roll(df_1w['close'].values, 1))
    tr_1w = np.maximum(tr_1w, np.roll(df_1w['high'].values, 1)) - np.minimum(df_1w['low'].values, np.roll(df_1w['high'].values, 1))
    tr_1w[0] = df_1w['high'][0] - df_1w['low'][0]  # first bar
    atr_1w = pd.Series(tr_1w).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    max_high_1w = pd.Series(df_1w['high'].values).rolling(window=14, min_periods=14).max().values
    min_low_1w = pd.Series(df_1w['low'].values).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(np.sum(atr_1w) / (np.log10(14) * (max_high_1w - min_low_1w)))
    chop = np.where((max_high_1w - min_low_1w) != 0, chop, 50)
    
    # Align Chop to 1d timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Warmup: max of KAMA seed (30), RSI (14), Chop (14)
    start_idx = 30
    
    for i in range(start_idx, n):
        kama_val = kama_1d_aligned[i]
        close_val = close[i]
        rsi_val = rsi_aligned[i]
        chop_val = chop_aligned[i]
        
        # Skip if any data not ready
        if (np.isnan(kama_val) or np.isnan(rsi_val) or np.isnan(chop_val)):
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Trend filter: price vs KAMA
        uptrend = close_val > kama_val
        downtrend = close_val < kama_val
        
        # RSI extremes: oversold < 30, overbought > 70
        rsi_oversold = rsi_val < 30
        rsi_overbought = rsi_val > 70
        
        # Chop regime: chop > 61.8 = range (mean revert), chop < 38.2 = trending (trend follow)
        chop_range = chop_val > 61.8
        chop_trending = chop_val < 38.2
        
        # Long: price > KAMA (uptrend) + RSI oversold + chop trending (to avoid false signals in strong chop)
        long_condition = uptrend and rsi_oversold and chop_trending
        # Short: price < KAMA (downtrend) + RSI overbought + chop trending
        short_condition = downtrend and rsi_overbought and chop_trending
        
        # Exit: opposite RSI extreme or chop becomes range
        long_exit = (position == 1 and (rsi_val > 70 or chop_range))
        short_exit = (position == -1 and (rsi_val < 30 or chop_range))
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
        elif long_exit:
            signals[i] = 0.0
            position = 0
        elif short_exit:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "1d_KAMA_Direction_RSI_ChopFilter_v1"
timeframe = "1d"
leverage = 1.0