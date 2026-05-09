#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_KAMA_RSI_Chop"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Choppiness index
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d Choppiness Index (CHOP)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(sum(trr14)/(hh-ll)) / log10(14)
    sum_tr = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    denominator = hh - ll
    chop = 100 * np.log10(sum_tr / denominator) / np.log10(14)
    
    chop = np.where(denominator == 0, 50, chop)  # avoid division by zero
    
    chop_12h = align_htf_to_ltf(prices, df_1d, chop)
    
    # 12h KAMA
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # Efficiency Ratio
    change = np.abs(np.diff(close_12h, k=10))  # 10-period change
    change = np.concatenate([[np.nan]*10, change])  # align
    
    volatility = np.abs(np.diff(close_12h))
    volatility = np.concatenate([[np.nan], volatility])
    vol_sum = pd.Series(volatility).rolling(window=10, min_periods=10).sum().values
    
    er = change / vol_sum
    er = np.where(vol_sum == 0, 0, er)
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2  # fast=2, slow=30
    kama = np.zeros_like(close_12h)
    kama[0] = close_12h[0]
    for i in range(1, len(close_12h)):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close_12h[i] - kama[i-1])
    
    kama_12h = align_htf_to_ltf(prices, df_12h, kama)
    
    # 12h RSI
    delta = np.diff(close_12h)
    delta = np.concatenate([[np.nan], delta])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    
    rs = avg_gain / avg_loss
    rs = np.where(avg_loss == 0, 0, rs)
    rsi = 100 - (100 / (1 + rs))
    rsi_12h = align_htf_to_ltf(prices, df_12h, rsi)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(chop_12h[i]) or np.isnan(kama_12h[i]) or 
            np.isnan(rsi_12h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        chop_val = chop_12h[i]
        kama_val = kama_12h[i]
        rsi_val = rsi_12h[i]
        price = close[i]
        
        # Chop filter: range when > 61.8, trend when < 38.2
        if chop_val > 61.8:  # ranging market - mean revert
            if position == 0:
                if price < kama_val and rsi_val < 40:
                    signals[i] = 0.25  # long
                    position = 1
                elif price > kama_val and rsi_val > 60:
                    signals[i] = -0.25  # short
                    position = -1
            elif position == 1:
                if price > kama_val or rsi_val > 50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                if price < kama_val or rsi_val < 50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
        else:  # trending market - follow trend
            if position == 0:
                if price > kama_val and rsi_val > 50:
                    signals[i] = 0.25  # long
                    position = 1
                elif price < kama_val and rsi_val < 50:
                    signals[i] = -0.25  # short
                    position = -1
            elif position == 1:
                if price < kama_val or rsi_val < 40:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                if price > kama_val or rsi_val > 60:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals