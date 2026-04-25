#!/usr/bin/env python3
"""
1d_KAMA_Direction_RSI_ChopFilter
Hypothesis: Trade 1d KAMA direction with RSI momentum filter and choppiness regime filter. KAMA adapts to market noise, reducing whipsaws. RSI ensures momentum alignment. Choppiness filter (CHOP>61.8) avoids trend-following in ranging markets. Works in bull/bear via adaptive trend detection + regime filter. Target 10-25 trades/year on 1d timeframe.
"""

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
    
    # Get 1d data for HTF indicators
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate KAMA on 1d
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close_1d, n=10))  # 10-period net change
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=1)  # 10-period sum of absolute changes
    # Handle volatility=0 case
    er = np.divide(change, volatility, out=np.zeros_like(change), where=volatility!=0)
    # Smoothing constants
    fastest = 2.0 / (2 + 1)   # for EMA=2
    slowest = 2.0 / (30 + 1)  # for EMA=30
    sc = (er * (fastest - slowest) + slowest) ** 2
    # Calculate KAMA
    kama = np.full_like(close_1d, np.nan, dtype=float)
    kama[9] = close_1d[9]  # start after 10 periods
    for i in range(10, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    kama_1d = kama
    
    # Calculate RSI(14) on 1d
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    rsi_1d = rsi
    
    # Calculate Choppiness Index (CHOP) on 1d
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # align with index
    # Sum of TR over 14 periods
    tr_sum_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    # Chop = 100 * log10(sum(tr14) / (hh14 - ll14)) / log10(14)
    range_14 = hh_14 - ll_14
    chop = np.full_like(close_1d, 100.0, dtype=float)
    mask = (range_14 > 0) & ~np.isnan(tr_sum_14)
    chop[mask] = 100 * np.log10(tr_sum_14[mask] / range_14[mask]) / np.log10(14)
    chop_1d = chop
    
    # Align HTF indicators to LTF (1d)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for KAMA (10), RSI (14), CHOP (14)
    start_idx = max(10, 14, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price > KAMA AND RSI > 50 AND chop > 61.8 (ranging market -> mean reversion to upside)
            long_setup = (close[i] > kama_aligned[i]) and \
                         (rsi_aligned[i] > 50) and \
                         (chop_aligned[i] > 61.8)
            # Short: price < KAMA AND RSI < 50 AND chop > 61.8 (ranging market -> mean reversion to downside)
            short_setup = (close[i] < kama_aligned[i]) and \
                          (rsi_aligned[i] < 50) and \
                          (chop_aligned[i] > 61.8)
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price < KAMA OR RSI < 40 (momentum loss)
            if (close[i] < kama_aligned[i]) or \
               (rsi_aligned[i] < 40):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price > KAMA OR RSI > 60 (momentum loss)
            if (close[i] > kama_aligned[i]) or \
               (rsi_aligned[i] > 60):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_KAMA_Direction_RSI_ChopFilter"
timeframe = "1d"
leverage = 1.0