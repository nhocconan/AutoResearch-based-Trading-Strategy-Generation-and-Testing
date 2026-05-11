#!/usr/bin/env python3
name = "12h_KAMA_Direction_RSI_Filter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for KAMA trend and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # KAMA (Kaufman Adaptive Moving Average) - trend direction
    # ER = Efficiency Ratio, SC = Smoothing Constant
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.abs(np.diff(close_1d))
    volatility_sum = np.zeros_like(volatility)
    for i in range(10, len(volatility)):
        volatility_sum[i] = np.sum(volatility[i-9:i+1])
    er = np.zeros_like(change)
    for i in range(len(change)):
        if volatility_sum[i] > 0:
            er[i] = change[i] / volatility_sum[i]
        else:
            er[i] = 0
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2  # fast=2, slow=30
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    kama_up = close_1d > kama
    
    # RSI(14) on 1d
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Get 1w data for regime filter (choppiness)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Choppiness Index (14) - range detection
    atr1w = np.zeros(len(high_1w))
    tr1w = np.zeros(len(high_1w))
    tr1w[0] = high_1w[0] - low_1w[0]
    for i in range(1, len(high_1w)):
        tr = high_1w[i] - low_1w[i]
        tr2 = abs(high_1w[i] - close_1w[i-1])
        tr3 = abs(low_1w[i] - close_1w[i-1])
        tr1w[i] = max(tr, tr2, tr3)
        atr1w[i] = np.mean(tr1w[max(0, i-13):i+1]) if i >= 14 else np.mean(tr1w[:i+1])
    
    highest_high = np.maximum.accumulate(high_1w)
    lowest_low = np.minimum.accumulate(low_1w)
    sum_atr14 = np.zeros_like(high_1w)
    for i in range(13, len(high_1w)):
        sum_atr14[i] = np.sum(atr1w[i-13:i+1])
    
    chop = np.zeros_like(high_1w)
    for i in range(14, len(high_1w)):
        if highest_high[i] - lowest_low[i] > 0:
            chop[i] = 100 * np.log10(sum_atr14[i] / (highest_high[i] - lowest_low[i])) / np.log10(14)
        else:
            chop[i] = 50
    
    # Align to 12h
    kama_up_aligned = align_htf_to_ltf(prices, df_1d, kama_up)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    
    # Volume confirmation on 12h
    vol_ma20 = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            if i > 0:
                vol_ma20[i] = np.mean(volume[:i+1])
        else:
            vol_ma20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 14)  # Need enough data
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(kama_up_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or
            np.isnan(chop_aligned[i]) or
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Chop regime: >61.8 = range (mean revert), <38.2 = trending (trend follow)
        is_ranging = chop_aligned[i] > 61.8
        is_trending = chop_aligned[i] < 38.2
        
        if position == 0:
            # Long: KAMA up + RSI > 50 (momentum) + volume confirmation
            if (kama_up_aligned[i] and 
                rsi_aligned[i] > 50 and 
                volume[i] > 1.2 * vol_ma20[i]):
                signals[i] = 0.25
                position = 1
            # Short: KAMA down + RSI < 50 + volume confirmation
            elif (not kama_up_aligned[i] and 
                  rsi_aligned[i] < 50 and 
                  volume[i] > 1.2 * vol_ma20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: KAMA down OR RSI < 40 (weakening) OR chop > 50 (ranging)
            if (not kama_up_aligned[i] or 
                rsi_aligned[i] < 40 or
                chop_aligned[i] > 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: KAMA up OR RSI > 60 (weakening) OR chop > 50 (ranging)
            if (kama_up_aligned[i] or 
                rsi_aligned[i] > 60 or
                chop_aligned[i] > 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals