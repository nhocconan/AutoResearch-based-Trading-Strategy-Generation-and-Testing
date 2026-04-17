#!/usr/bin/env python3
"""
12h_KAMA_RSI_Chop_v1
KAMA trend direction + RSI filter + Choppy market filter.
Long when KAMA bullish, RSI > 50, and Chop < 61.8 (trending).
Short when KAMA bearish, RSI < 50, and Chop < 61.8 (trending).
Exit when conditions reverse or Chop > 61.8 (ranging).
Uses 1d trend filter: price above/below 1d EMA200 for long/short bias.
Designed to capture trending moves while avoiding choppy markets.
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # === KAMA(10) ===
    change = np.abs(close - np.roll(close, 1))
    change[0] = 0
    direction = np.abs(close - np.roll(close, 10))
    direction[0:10] = 0
    er = np.where(change != 0, direction / np.nansum(change.reshape(-1, 10), axis=1), 0)
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # seed
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === RSI(14) ===
    delta = np.diff(close)
    delta = np.concatenate([[0], delta])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # === Choppy Market Index(14) ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (hh - ll + 1e-10)) / np.log10(14)
    
    # === 1d EMA200 for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    ema_200_1d = pd.Series(df_1d['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(chop[i]) or 
            np.isnan(ema_200_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: KAMA rising, RSI > 50, Chop < 61.8 (trending), price above 1d EMA200
            if (kama[i] > kama[i-1] and 
                rsi[i] > 50 and 
                chop[i] < 61.8 and 
                close[i] > ema_200_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: KAMA falling, RSI < 50, Chop < 61.8 (trending), price below 1d EMA200
            elif (kama[i] < kama[i-1] and 
                  rsi[i] < 50 and 
                  chop[i] < 61.8 and 
                  close[i] < ema_200_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: KAMA falling OR RSI < 50 OR Chop > 61.8 (ranging)
            if (kama[i] < kama[i-1] or 
                rsi[i] < 50 or 
                chop[i] > 61.8):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: KAMA rising OR RSI > 50 OR Chop > 61.8 (ranging)
            if (kama[i] > kama[i-1] or 
                rsi[i] > 50 or 
                chop[i] > 61.8):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_KAMA_RSI_Chop_v1"
timeframe = "12h"
leverage = 1.0