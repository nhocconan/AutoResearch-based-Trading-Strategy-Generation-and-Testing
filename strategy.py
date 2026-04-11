#!/usr/bin/env python3
# 1d_1w_kama_rsi_chop_v1
# Strategy: 1d KAMA direction with RSI filter and Chop regime
# Timeframe: 1d
# Leverage: 1.0
# Hypothesis: KAMA adapts to trend changes, RSI filters extremes, Chop filter avoids whipsaw in range.
# Works in bull via KAMA up + RSI<70, bear via KAMA down + RSI>30. Chop>61.8 avoids false signals in range.
# Designed for low trade frequency (~10-25/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_kama_rsi_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1d KAMA (ER=10)
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close))
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # 1d RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # 1d Chop(14)
    atr = np.zeros(n)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10((max_high - min_low) / (atr * 14)) / np.log10(14)
    
    # 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or np.isnan(ema_50_1w_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Chop filter: Chop > 61.8 = range (avoid trend signals)
        chop_filter = chop[i] > 61.8
        
        # KAMA direction
        kama_up = close[i] > kama[i]
        kama_down = close[i] < kama[i]
        
        # RSI filter: avoid extremes
        rsi_not_overbought = rsi[i] < 70
        rsi_not_oversold = rsi[i] > 30
        
        # Entry conditions
        # Long: KAMA up AND RSI not overbought AND Chop filter AND price above 1w EMA
        if kama_up and rsi_not_overbought and chop_filter and close[i] > ema_50_1w_aligned[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: KAMA down AND RSI not oversold AND Chop filter AND price below 1w EMA
        elif kama_down and rsi_not_oversold and chop_filter and close[i] < ema_50_1w_aligned[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: Opposite KAMA signal
        elif position == 1 and kama_down:
            position = 0
            signals[i] = 0.0
        elif position == -1 and kama_up:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals