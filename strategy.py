#!/usr/bin/env python3
# 12h_1d_kama_rsi_chop_v1
# Strategy: 12h KAMA trend with 1d RSI and Choppiness index filter
# Timeframe: 12h
# Leverage: 1.0
# Hypothesis: KAMA adapts to market noise, RSI identifies overbought/oversold, Choppiness identifies ranging vs trending.
# Long when KAMA rising, RSI < 40, and CHOP > 61.8 (ranging). Short when KAMA falling, RSI > 60, and CHOP > 61.8.
# Designed for low frequency (15-25 trades/year) to minimize fee drag in ranging markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_kama_rsi_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d RSI(14)
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # 1d Choppiness Index(14)
    atr_1d = np.zeros(len(close_1d))
    tr1 = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    tr2 = np.abs(np.diff(high_1d := df_1d['high'].values, prepend=high_1d[0]))
    tr3 = np.abs(np.diff(low_1d := df_1d['low'].values, prepend=low_1d[0]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(atr_1d * 14 / (highest_high - lowest_low + 1e-10)) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # 12h KAMA
    # Efficiency Ratio
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0) if False else None  # placeholder
    # Correct ER calculation
    er = np.zeros(len(close))
    for i in range(10, len(close)):  # ER needs 10-period lookback
        if i >= 10:
            net_change = abs(close[i] - close[i-10])
            total_change = np.sum(np.abs(np.diff(close[i-10:i+1])))
            er[i] = net_change / (total_change + 1e-10)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    kama = np.zeros(len(close))
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(rsi_1d_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(kama[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # KAMA direction
        kama_rising = kama[i] > kama[i-1]
        kama_falling = kama[i] < kama[i-1]
        
        # RSI levels
        rsi = rsi_1d_aligned[i]
        rsi_oversold = rsi < 40
        rsi_overbought = rsi > 60
        
        # Choppiness: > 61.8 = ranging (good for mean reversion)
        chop = chop_aligned[i]
        ranging = chop > 61.8
        
        # Entry logic: KAMA direction + RSI extreme + ranging market
        if (kama_rising and rsi_oversold and ranging and position != 1):
            position = 1
            signals[i] = 0.25
        elif (kama_falling and rsi_overbought and ranging and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: KAMA direction change or market becomes trending
        elif position == 1 and (not kama_rising or not ranging):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (not kama_falling or not ranging):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals