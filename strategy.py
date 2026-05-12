#/usr/bin/env python3
"""
12h_KAMA_Trend_With_Volume_And_Chop_Filter
Hypothesis: Uses 1d KAMA direction as primary trend filter, enters on 12h price crossing above/below KAMA with volume confirmation and choppiness regime filter to avoid whipsaws. Designed for low trade frequency (20-50/year) to minimize fee drift while capturing trends in both bull and bear markets.
"""
name = "12h_KAMA_Trend_With_Volume_And_Chop_Filter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume spike: >1.5x 20-period average (on 12h timeframe)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Daily KAMA for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # KAMA parameters: ER decay fast/slow
    er = np.zeros_like(close_1d)
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=0)  # placeholder, will compute properly below
    # Recompute volatility as sum of absolute changes over 10 periods
    volatility = pd.Series(change).rolling(window=10, min_periods=10).sum().values
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # 12h Choppiness Index for regime filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = np.abs(high_12h[1:] - low_12h[1:])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    max_high = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    
    chop = np.where((max_high - min_low) != 0, 
                    -100 * np.log10(np.sum(tr, axis=0) / (max_high - min_low)) / np.log10(14), 50)
    # Fix chop calculation: sum over window
    chop = np.full_like(close_12h, np.nan)
    for i in range(14, len(close_12h)):
        tr_sum = np.nansum(tr[i-13:i+1])
        hh = np.nanmax(high_12h[i-13:i+1])
        ll = np.nanmin(low_12h[i-13:i+1])
        if hh - ll != 0:
            chop[i] = -100 * np.log10(tr_sum / (hh - ll)) / np.log10(14)
        else:
            chop[i] = 50
    
    # Align all indicators to 12h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    chop_aligned = align_htf_to_ltf(prices, df_12h, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if (np.isnan(kama_aligned[i]) or
            np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price above KAMA + chop < 61.8 (trending) + volume spike
            if (close[i] > kama_aligned[i] and 
                chop_aligned[i] < 61.8 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price below KAMA + chop < 61.8 (trending) + volume spike
            elif (close[i] < kama_aligned[i] and 
                  chop_aligned[i] < 61.8 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price below KAMA OR chop > 61.8 (choppy)
            if (close[i] < kama_aligned[i]) or \
               (chop_aligned[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price above KAMA OR chop > 61.8 (choppy)
            if (close[i] > kama_aligned[i]) or \
               (chop_aligned[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals