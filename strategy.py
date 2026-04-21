#!/usr/bin/env python3
"""
12h_1d_KAMA_Trend_With_Chop_Filter
Hypothesis: KAMA trend direction from 1d timeframe combined with 12h Choppiness Index filter.
Long when KAMA is rising and CHOP > 61.8 (range) for mean reversion to upside.
Short when KAMA is falling and CHOP > 61.8 for mean reversion to downside.
Uses volume confirmation to avoid false signals.
Designed for 12h timeframe to capture multi-day mean reversion moves in ranging markets.
Works in both bull and bear markets by adapting to regime via Choppiness Index.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data once for KAMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # KAMA calculation
    close_s = pd.Series(close_1d)
    # Efficiency Ratio
    change = abs(close_s.diff(10))
    volatility = close_s.diff().abs().rolling(10, min_periods=1).sum()
    er = change / volatility.replace(0, np.nan)
    er = er.fillna(0)
    # Smoothing constants
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2
    # KAMA
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Shift KAMA to align with current day (no look-ahead)
    kama = np.roll(kama, 1)
    kama[0] = np.nan
    
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # Calculate Choppiness Index on 12h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR(14)
    atr = np.zeros(n)
    atr[13] = np.nanmean(tr[1:14])  # First valid ATR at index 13
    for i in range(14, n):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Sum of ATR over 14 periods
    sum_tr14 = np.zeros(n)
    for i in range(13, n):
        if i == 13:
            sum_tr14[i] = np.nansum(tr[1:14])
        else:
            sum_tr14[i] = sum_tr14[i-1] - tr[i-13] + tr[i]
    
    # Choppiness Index
    chop = np.full(n, np.nan)
    for i in range(13, n):
        if sum_tr14[i] > 0:
            max_high = np.max(high[i-13:i+1])
            min_low = np.min(low[i-13:i+1])
            chop[i] = 100 * np.log10(sum_tr14[i] / (max_high - min_low)) / np.log10(14)
    
    # Volume confirmation: current volume > 1.3 * 20-period average
    volume = prices['volume'].values
    vol_ma = np.zeros(n)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(chop[i]) or 
            i < 20 or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol_ok = volume[i] > 1.3 * vol_ma[i]
        
        if position == 0:
            # Long: KAMA rising (trend up) AND choppy market (mean reversion long)
            if kama_aligned[i] > kama_aligned[i-1] and chop[i] > 61.8 and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: KAMA falling (trend down) AND choppy market (mean reversion short)
            elif kama_aligned[i] < kama_aligned[i-1] and chop[i] > 61.8 and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: KAMA starts falling OR market becomes trending
            if kama_aligned[i] < kama_aligned[i-1] or chop[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: KAMA starts rising OR market becomes trending
            if kama_aligned[i] > kama_aligned[i-1] or chop[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1d_KAMA_Trend_With_Chop_Filter"
timeframe = "12h"
leverage = 1.0