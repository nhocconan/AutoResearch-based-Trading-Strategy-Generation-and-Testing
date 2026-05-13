#!/usr/bin/env python3
name = "1d_KAMA_RSI_Chop_Reversal_v1"
timeframe = "1d"
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
    
    # KAMA parameters
    er_len = 10
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=er_len))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    er = np.zeros(n)
    for i in range(er_len, n):
        if volatility[i] != 0:
            er[i] = change[i - er_len + 1:i + 1].sum() / volatility[i]
        else:
            er[i] = 0
    
    # Smoothing constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI
    rsi_len = 14
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    avg_gain[rsi_len] = np.mean(gain[1:rsi_len+1])
    avg_loss[rsi_len] = np.mean(loss[1:rsi_len+1])
    for i in range(rsi_len+1, n):
        avg_gain[i] = (avg_gain[i-1] * (rsi_len-1) + gain[i]) / rsi_len
        avg_loss[i] = (avg_loss[i-1] * (rsi_len-1) + loss[i]) / rsi_len
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index (weekly)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    chop_len = 14
    atr_1w = np.zeros(len(high_1w))
    for i in range(1, len(high_1w)):
        tr = max(high_1w[i] - low_1w[i], 
                 abs(high_1w[i] - close_1w[i-1]), 
                 abs(low_1w[i] - close_1w[i-1]))
        atr_1w[i] = tr
    
    sum_atr = np.zeros(len(high_1w))
    for i in range(chop_len, len(high_1w)):
        sum_atr[i] = np.sum(atr_1w[i-chop_len+1:i+1])
    
    max_range = np.zeros(len(high_1w))
    for i in range(chop_len, len(high_1w)):
        max_range[i] = np.max(high_1w[i-chop_len+1:i+1]) - np.min(low_1w[i-chop_len+1:i+1])
    
    chop = np.zeros(len(high_1w))
    for i in range(chop_len, len(high_1w)):
        if max_range[i] != 0:
            chop[i] = 100 * np.log10(sum_atr[i] / max_range[i]) / np.log10(chop_len)
        else:
            chop[i] = 50
    
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    
    # Signals
    signals = np.zeros(n)
    position = 0
    
    for i in range(50, n):
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or 
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Chop regime: > 61.8 = range (mean revert), < 38.2 = trending
        if chop_aligned[i] > 61.8:  # Range regime - mean revert
            if position == 0:
                if close[i] < kama[i] and rsi[i] < 30:
                    signals[i] = 0.25
                    position = 1
                elif close[i] > kama[i] and rsi[i] > 70:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            elif position == 1:
                if close[i] > kama[i] or rsi[i] > 50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                if close[i] < kama[i] or rsi[i] < 50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
        else:  # Trending regime - follow trend
            if position == 0:
                if close[i] > kama[i] and rsi[i] > 50:
                    signals[i] = 0.25
                    position = 1
                elif close[i] < kama[i] and rsi[i] < 50:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            elif position == 1:
                if close[i] < kama[i] or rsi[i] < 50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                if close[i] > kama[i] or rsi[i] > 50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals