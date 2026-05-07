#!/usr/bin/env python3
name = "1d_KAMA_RSI_Chop_Filter_v2"
timeframe = "1d"
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
    
    # 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w KAMA trend filter
    close_1w = df_1w['close'].values
    change_1w = np.abs(np.diff(close_1w, prepend=close_1w[0]))
    dir_1w = np.abs(np.diff(close_1w, k=10, prepend=close_1w[:10]))
    volatility_1w = np.sum(change_1w[1:], axis=0) if len(change_1w) > 1 else np.zeros_like(close_1w)
    er_1w = np.where(volatility_1w != 0, dir_1w / volatility_1w, 0)
    sc_1w = (er_1w * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama_1w = np.zeros_like(close_1w)
    kama_1w[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        kama_1w[i] = kama_1w[i-1] + sc_1w[i] * (close_1w[i] - kama_1w[i-1])
    kama_1w_ma = pd.Series(kama_1w).ewm(span=10, adjust=False, min_periods=10).mean().values
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w_ma)
    
    # RSI(14) on daily
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index (14) on daily
    atr1 = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    atr1[0] = high[0] - low[0]
    atr_sum = pd.Series(atr1).rolling(window=14, min_periods=14).sum().values
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (hh - ll)) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(kama_1w_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above weekly KAMA + RSI > 50 + chop < 61.8 (trending)
            if close[i] > kama_1w_aligned[i] and rsi[i] > 50 and chop[i] < 61.8:
                signals[i] = 0.25
                position = 1
            # Short: price below weekly KAMA + RSI < 50 + chop < 61.8 (trending)
            elif close[i] < kama_1w_aligned[i] and rsi[i] < 50 and chop[i] < 61.8:
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: trend fails or chop indicates range
            if position == 1:
                if close[i] < kama_1w_aligned[i] or chop[i] > 61.8:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > kama_1w_aligned[i] or chop[i] > 61.8:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals