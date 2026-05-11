#!/usr/bin/env python3
name = "4h_KAMA_RSI_ChopFilter_Scaled"
timeframe = "4h"
leverage = 1.0

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
    volume = prices['volume'].values
    
    # Get 1d data for Chop filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Chop filter: high/low over 14-day period
    high_14 = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    low_14 = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    atr_14 = pd.Series(high_14 - low_14).rolling(window=14, min_periods=14).mean().values
    chop_num = pd.Series(close).rolling(window=14, min_periods=14).sum().values
    chop_denom = atr_14 * 14
    chop = 100 * np.log10(chop_num / chop_denom) / np.log10(14)
    chop = np.where(chop_denom > 0, chop, 50)  # avoid div by zero
    chop_filter = chop > 61.8  # chop > 61.8 = ranging (mean reversion)
    chop_filter_aligned = align_htf_to_ltf(prices, df_1d, chop_filter)
    
    # KAMA on 4h close
    change = np.abs(np.diff(close, periods=10))
    volatility = np.sum(np.abs(np.diff(close)), axis=1)
    er = np.where(volatility > 0, change / volatility, 0)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI on 4h close
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if np.isnan(chop_filter_aligned[i]) or np.isnan(kama[i]) or np.isnan(rsi[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: KAMA up, RSI < 30, chop > 61.8 (range)
            if kama[i] > kama[i-1] and rsi[i] < 30 and chop_filter_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: KAMA down, RSI > 70, chop > 61.8 (range)
            elif kama[i] < kama[i-1] and rsi[i] > 70 and chop_filter_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: KAMA down or RSI > 70
            if kama[i] < kama[i-1] or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: KAMA up or RSI < 30
            if kama[i] > kama[i-1] or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals