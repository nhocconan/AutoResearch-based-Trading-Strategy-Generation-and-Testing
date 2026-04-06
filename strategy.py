#!/usr/bin/env python3
"""
4h KAMA Direction + RSI + Chop Filter
Hypothesis: KAMA adapts to market noise, providing reliable trend direction.
In trending markets, price follows KAMA direction. Choppy markets filtered by Choppiness Index.
Long when KAMA rising and RSI > 50, short when KAMA falling and RSI < 50.
Chop filter prevents entries in ranging markets (CHOP > 61.8). Works in both bull and bear.
Target: 75-200 total trades over 4 years (19-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14323_4h_kama_rsi_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate KAMA for daily trend filter
    kama_period = 10
    fast_ema = 2
    slow_ema = 30
    
    # Efficiency Ratio
    change = np.abs(np.diff(close_1d, kama_period))
    abs_change = np.sum(np.abs(np.diff(close_1d)), axis=0) if len(close_1d) > 1 else 0
    # Simplified ER calculation for array
    er = np.zeros_like(close_1d)
    for i in range(kama_period, len(close_1d)):
        if np.sum(np.abs(np.diff(close_1d[i-kama_period:i+1]))) > 0:
            er[i] = np.abs(close_1d[i] - close_1d[i-kama_period]) / np.sum(np.abs(np.diff(close_1d[i-kama_period:i+1])))
    
    # Smoothing constant
    sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2
    kama = np.zeros_like(close_1d)
    kama[kama_period] = close_1d[kama_period]
    for i in range(kama_period+1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    kama_1d = kama
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # 4h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # RSI (14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index (14)
    atr_period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_sum = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).sum().values
    highest_high = pd.Series(high).rolling(window=atr_period, min_periods=atr_period).max().values
    lowest_low = pd.Series(low).rolling(window=atr_period, min_periods=atr_period).min().values
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(atr_period)
    
    # ATR for stoploss
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (0.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(30, 14, 20) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(kama_1d_aligned[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: KAMA turns down OR RSI < 40 OR stoploss
            if kama_1d_aligned[i] < kama_1d_aligned[i-1] or rsi[i] < 40 or close[i] <= entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: KAMA turns up OR RSI > 60 OR stoploss
            if kama_1d_aligned[i] > kama_1d_aligned[i-1] or rsi[i] > 60 or close[i] >= entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: KAMA direction + RSI + chop filter + volume
            kama_rising = kama_1d_aligned[i] > kama_1d_aligned[i-1]
            kama_falling = kama_1d_aligned[i] < kama_1d_aligned[i-1]
            long_setup = kama_rising and (rsi[i] > 50) and (chop[i] < 61.8) and vol_filter[i]
            short_setup = kama_falling and (rsi[i] < 50) and (chop[i] < 61.8) and vol_filter[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_setup:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals