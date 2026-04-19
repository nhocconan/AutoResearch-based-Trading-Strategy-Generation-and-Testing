#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h KAMA + RSI + Choppiness regime filter (1d)
# Uses KAMA for adaptive trend detection, RSI for momentum, and 1d Choppiness Index to filter range-bound markets.
# KAMA adapts to market efficiency - slow in chop, fast in trends.
# Long when KAMA rising, RSI > 50, and CHOP < 38.2 (trending regime).
# Short when KAMA falling, RSI < 50, and CHOP < 38.2.
# Exit when KAMA direction reverses or CHOP > 61.8 (choppy regime).
# Target: 20-30 trades/year per symbol with disciplined entries.
name = "4h_KAMA_RSI_ChopFilter_1d"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d Choppiness Index for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # True Range
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Highest high and lowest low over 14 periods
    hh_1d = df_1d['high'].rolling(window=14, min_periods=14).max().values
    ll_1d = df_1d['low'].rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(sum(TR14) / (HH14 - LL14)) / log10(14)
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_tr_14 / (hh_1d - ll_1d)) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # KAMA (Kaufman Adaptive Moving Average)
    def kama(close, er_len=10, fast=2, slow=30):
        # Efficiency Ratio
        change = np.abs(np.diff(close, n=er_len))
        volatility = np.sum(np.abs(np.diff(close)), axis=0)
        er = np.zeros_like(close)
        for i in range(er_len, len(close)):
            if volatility[i] != 0:
                er[i] = change[i-er_len] / volatility[i]
            else:
                er[i] = 0
        
        # Smoothing constants
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        
        # KAMA calculation
        kama_val = np.zeros_like(close)
        kama_val[0] = close[0]
        for i in range(1, len(close)):
            kama_val[i] = kama_val[i-1] + sc[i] * (close[i] - kama_val[i-1])
        return kama_val
    
    kama_val = kama(close, er_len=10, fast=2, slow=30)
    
    # RSI (14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    # Pad RSI to match length (first 14 values are NaN)
    rsi_padded = np.full_like(close, np.nan, dtype=float)
    rsi_padded[14:] = rsi
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_val[i]) or np.isnan(rsi_padded[i]) or 
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # KAMA direction: rising if current > previous, falling if current < previous
        kama_rising = i > 0 and kama_val[i] > kama_val[i-1]
        kama_falling = i > 0 and kama_val[i] < kama_val[i-1]
        
        if position == 0:
            # Long: KAMA rising, RSI > 50, and trending regime (CHOP < 38.2)
            if (kama_rising and rsi_padded[i] > 50 and chop_aligned[i] < 38.2):
                signals[i] = 0.25
                position = 1
            # Short: KAMA falling, RSI < 50, and trending regime (CHOP < 38.2)
            elif (kama_falling and rsi_padded[i] < 50 and chop_aligned[i] < 38.2):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if KAMA falls or regime turns choppy (CHOP > 61.8)
            if (not kama_rising) or (chop_aligned[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if KAMA rises or regime turns choppy (CHOP > 61.8)
            if (not kama_falling) or (chop_aligned[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals