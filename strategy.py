#!/usr/bin/env python3
name = "1d_KAMA_Direction_RSI_Chop_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # KAMA parameters
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1) # EMA(30)
    
    # Calculate Efficiency Ratio and Smoothing Constant
    change = np.abs(np.diff(close, n=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0)  # 10-period volatility
    
    # Vectorized ER calculation
    er = np.zeros(n)
    er[10:] = change[10:] / np.maximum(volatility[10:], 1e-10)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    weekly_up = close > ema_50_1w_aligned
    weekly_down = close < ema_50_1w_aligned
    
    # Daily RSI
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / np.maximum(avg_loss, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Weekly Choppiness Index
    df_1w_chop = get_htf_data(prices, '1w')
    if len(df_1w_chop) < 14:
        return np.zeros(n)
    
    high_1w = df_1w_chop['high'].values
    low_1w = df_1w_chop['low'].values
    close_1w = df_1w_chop['close'].values
    
    atr_1w = np.zeros(len(high_1w))
    tr1 = np.abs(high_1w[1:] - low_1w[1:])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1w[1:] = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    max_high = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    
    chop = np.zeros(len(high_1w))
    for i in range(14, len(high_1w)):
        if atr_1w[i] > 0 and (max_high[i] - min_low[i]) > 0:
            chop[i] = 100 * np.log10(np.sum(atr_1w[i-13:i+1]) / np.maximum(max_high[i] - min_low[i], 1e-10)) / np.log10(14)
        else:
            chop[i] = 50
    
    chop_aligned = align_htf_to_ltf(prices, df_1w_chop, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 14)  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: KAMA up, weekly up, RSI > 50, chop < 61.8 (trending)
            if (close[i] > kama[i] and 
                weekly_up[i] and 
                rsi[i] > 50 and 
                chop_aligned[i] < 61.8):
                signals[i] = 0.25
                position = 1
            # Short: KAMA down, weekly down, RSI < 50, chop < 61.8 (trending)
            elif (close[i] < kama[i] and 
                  weekly_down[i] and 
                  rsi[i] < 50 and 
                  chop_aligned[i] < 61.8):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: KAMA down OR weekly down OR RSI < 40 OR chop > 61.8 (choppy)
            if (close[i] < kama[i] or 
                not weekly_up[i] or 
                rsi[i] < 40 or 
                chop_aligned[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: KAMA up OR weekly up OR RSI > 60 OR chop > 61.8 (choppy)
            if (close[i] > kama[i] or 
                not weekly_down[i] or 
                rsi[i] > 60 or 
                chop_aligned[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: KAMA direction with weekly trend filter, RSI momentum, and chop regime filter
# captures trends in both bull and bear markets while avoiding whipsaws in choppy conditions.
# KAMA adapts to market efficiency, providing timely trend signals.
# Weekly trend filter ensures alignment with higher timeframe momentum.
# RSI filters for momentum strength, chop filter avoids ranging markets.
# This combination should work in BTC/ETH across market regimes with controlled trade frequency.