#!/usr/bin/env python3
name = "4h_KAMA_Direction_RSI_Chop"
timeframe = "4h"
leverage = 1.0

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
    
    # === 1D DATA FOR TREND FILTER ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Daily KAMA for trend direction
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    direction = np.abs(np.diff(close_1d, k=10, prepend=close_1d[:10]))
    er = np.where(direction != 0, change / direction, 0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # === RSI(14) FOR MOMENTUM ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # === CHOPPINESS INDEX (14) FOR REGIME ===
    atr = np.zeros(n)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(np.sum(tr, axis=0) / (max_high - min_low)) / np.log10(14) if False else \
           100 * np.log10(pd.Series(tr).rolling(window=14, min_periods=14).sum().values / (max_high - min_low)) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 14)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama_1d_aligned[i]) or np.isnan(rsi[i]) or np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price above daily KAMA, RSI > 50, Chop < 61.8 (trending)
            if (close[i] > kama_1d_aligned[i] and 
                rsi[i] > 50 and 
                chop[i] < 61.8):
                signals[i] = 0.25
                position = 1
            # SHORT: Price below daily KAMA, RSI < 50, Chop < 61.8 (trending)
            elif (close[i] < kama_1d_aligned[i] and 
                  rsi[i] < 50 and 
                  chop[i] < 61.8):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price crosses below daily KAMA OR RSI < 40
            if (close[i] < kama_1d_aligned[i]) or (rsi[i] < 40):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above daily KAMA OR RSI > 60
            if (close[i] > kama_1d_aligned[i]) or (rsi[i] > 60):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals