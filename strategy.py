#!/usr/bin/env python3
name = "1d_KAMA_RSI_ChopFilter_v1"
timeframe = "1d"
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
    
    # Get weekly data
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # KAMA on 1W for trend
    close_1w = df_1w['close'].values
    change = np.abs(np.diff(close_1w, prepend=close_1w[0]))
    volatility = np.abs(np.diff(close_1w))
    er = np.zeros_like(close_1w)
    er[0] = 0
    for i in range(1, len(close_1w)):
        if volatility[i] != 0:
            er[i] = change[i] / volatility[i]
        else:
            er[i] = 1
    sc = (er * 0.29 + 0.06) ** 2
    kama = np.zeros_like(close_1w)
    kama[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        kama[i] = kama[i-1] + sc[i] * (close_1w[i] - kama[i-1])
    kama_1w = kama
    
    # Align KAMA
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    
    # Chop on 1D
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_period = 14
    atr = np.zeros(n)
    atr[:atr_period] = np.nan
    for i in range(atr_period, n):
        atr[i] = np.nanmean(tr[i-atr_period+1:i+1])
    sum_tr = np.zeros(n)
    for i in range(n):
        if i < atr_period:
            sum_tr[i] = np.sum(tr[max(0, i-atr_period+1):i+1]) if i > 0 else 0
        else:
            sum_tr[i] = np.sum(tr[i-atr_period+1:i+1])
    hh = np.zeros(n)
    ll = np.zeros(n)
    for i in range(n):
        if i < atr_period:
            hh[i] = np.max(high[max(0, i-atr_period+1):i+1]) if i > 0 else high[i]
            ll[i] = np.min(low[max(0, i-atr_period+1):i+1]) if i > 0 else low[i]
        else:
            hh[i] = np.max(high[i-atr_period+1:i+1])
            ll[i] = np.min(low[i-atr_period+1:i+1])
    chop = np.zeros(n)
    for i in range(n):
        if hh[i] != ll[i] and not np.isnan(sum_tr[i]):
            chop[i] = 100 * np.log10(sum_tr[i] / (hh[i] - ll[i])) / np.log10(atr_period)
        else:
            chop[i] = np.nan
    
    # RSI on 1D
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    for i in range(n):
        if i < 14:
            avg_gain[i] = np.mean(gain[max(0, i-13):i+1]) if i > 0 else 0
            avg_loss[i] = np.mean(loss[max(0, i-13):i+1]) if i > 0 else 0
        else:
            avg_gain[i] = np.mean(gain[i-13:i+1])
            avg_loss[i] = np.mean(loss[i-13:i+1])
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 14)
    
    for i in range(start_idx, n):
        if (np.isnan(kama_1w_aligned[i]) or np.isnan(chop[i]) or np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Long: price > KAMA, RSI > 50, Chop < 61.8 (trending)
        if close[i] > kama_1w_aligned[i] and rsi[i] > 50 and chop[i] < 61.8:
            if position == 0:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        # Short: price < KAMA, RSI < 50, Chop < 61.8 (trending)
        elif close[i] < kama_1w_aligned[i] and rsi[i] < 50 and chop[i] < 61.8:
            if position == 0:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = -0.25
        else:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
    
    return signals