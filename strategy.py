#!/usr/bin/env python3
"""
1d KAMA + RSI + Chop Filter
Trades only when price aligns with KAMA direction, RSI confirms momentum,
and Chop confirms ranging regime (mean reversion). Works in bull/bear by
avoiding strong trends and capturing mean reversion in ranges.
"""
name = "1d_KAMA_RSI_Chop_Filter"
timeframe = "1d"
leverage = 1.0

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
    
    # === WEEKLY DATA FOR TREND FILTER ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Weekly KAMA
    # ER: Efficiency Ratio
    change = np.abs(np.diff(close_1w, prepend=close_1w[0]))
    volatility = np.sum(np.abs(np.diff(close_1w)), axis=0)
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (0.6667 - 0.0645) + 0.0645) ** 2
    kama = np.zeros_like(close_1w)
    kama[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        kama[i] = kama[i-1] + sc[i] * (close_1w[i] - kama[i-1])
    kama_1w = kama
    
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    
    # Weekly Chop
    atr_1w = np.zeros_like(close_1w)
    tr1 = np.abs(high_1w - low_1w)
    tr2 = np.abs(np.subtract(high_1w, np.roll(close_1w, 1)))
    tr3 = np.abs(np.subtract(low_1w, np.roll(close_1w, 1)))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    sum_tr = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    chop = np.where((max_high - min_low) != 0, 100 * np.log10(sum_tr / (max_high - min_low)) / np.log10(14), 50)
    chop_1w = chop
    
    chop_1w_aligned = align_htf_to_ltf(prices, df_1w, chop_1w)
    
    # === DAILY INDICATORS ===
    # Daily RSI
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Daily volume MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 14  # for weekly ATR and Chop
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama_1w_aligned[i]) or np.isnan(chop_1w_aligned[i]) or 
            np.isnan(rsi[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price above weekly KAMA, RSI > 50, Chop < 50 (trending up)
            if (close[i] > kama_1w_aligned[i] and 
                rsi[i] > 50 and 
                chop_1w_aligned[i] < 50):
                signals[i] = 0.25
                position = 1
            # SHORT: Price below weekly KAMA, RSI < 50, Chop < 50 (trending down)
            elif (close[i] < kama_1w_aligned[i] and 
                  rsi[i] < 50 and 
                  chop_1w_aligned[i] < 50):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price crosses below weekly KAMA or Chop > 60 (ranging)
            if (close[i] < kama_1w_aligned[i]) or (chop_1w_aligned[i] > 60):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above weekly KAMA or Chop > 60 (ranging)
            if (close[i] > kama_1w_aligned[i]) or (chop_1w_aligned[i] > 60):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals