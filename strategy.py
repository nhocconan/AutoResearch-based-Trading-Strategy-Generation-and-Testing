#!/usr/bin/env python3
"""
1d KAMA + RSI + Chop Filter
Long: KAMA rising + RSI < 50 + Chop > 61.8 (range)
Short: KAMA falling + RSI > 50 + Chop > 61.8 (range)
Exit: Opposite KAMA direction change
Targets choppy markets where mean reversion works, avoids strong trends
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 1d KAMA (ER=10)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close']
    change = abs(close_1d.diff(10))
    volatility = close_1d.diff().abs().rolling(10).sum()
    er = change / volatility.replace(0, np.nan)
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2
    kama = [close_1d.iloc[0]]
    for i in range(1, len(close_1d)):
        kama.append(kama[-1] + sc.iloc[i] * (close_1d.iloc[i] - kama[-1]))
    kama = np.array(kama)
    kama_1d = align_htf_to_ltf(prices, df_1d, kama)
    
    # 1d Chop (14-period)
    atr1 = df_1d['high'] - df_1d['low']
    atr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    atr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = np.maximum(atr1, np.maximum(atr2, atr3))
    atr = pd.Series(tr).rolling(14, min_periods=14).mean().values
    max_hh = df_1d['high'].rolling(14, min_periods=14).max().values
    min_ll = df_1d['low'].rolling(14, min_periods=14).min().values
    chop = 100 * np.log10(np.sum(atr) / (max_hh - min_ll)) / np.log10(14)
    chop = np.where((max_hh - min_ll) == 0, 50, chop)
    chop_1d = align_htf_to_ltf(prices, df_1d, chop)
    
    # 1d RSI (14)
    delta = pd.Series(close_1d).diff()
    gain = delta.where(delta > 0, 0).rolling(14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    rsi_1d = align_htf_to_ltf(prices, df_1d, rsi)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(kama_1d[i]) or np.isnan(chop_1d[i]) or 
            np.isnan(rsi_1d[i])):
            signals[i] = 0.0
            continue
        
        kama_val = kama_1d[i]
        chop_val = chop_1d[i]
        rsi_val = rsi_1d[i]
        
        if position == 0:
            # Long: KAMA rising + RSI < 50 + Chop > 61.8
            if i > 0 and kama_val > kama_1d[i-1] and rsi_val < 50 and chop_val > 61.8:
                signals[i] = 0.25
                position = 1
            # Short: KAMA falling + RSI > 50 + Chop > 61.8
            elif i > 0 and kama_val < kama_1d[i-1] and rsi_val > 50 and chop_val > 61.8:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: KAMA falls
            if i > 0 and kama_val < kama_1d[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: KAMA rises
            if i > 0 and kama_val > kama_1d[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_RSI_Chop"
timeframe = "1d"
leverage = 1.0