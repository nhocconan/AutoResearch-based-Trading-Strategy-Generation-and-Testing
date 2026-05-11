#!/usr/bin/env python3
name = "1d_KAMA_RSI_ChopFilter_Trend"
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
    
    # Get 1d data for KAMA and RSI (using daily timeframe)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) on 1d data
    close_1d = df_1d['close'].values
    # Efficiency ratio (ER) = |price change over period| / sum of absolute price changes
    change = np.abs(np.diff(close_1d))
    volatility = np.sum(change)
    if volatility > 0:
        er = np.abs(close_1d[-1] - close_1d[0]) / volatility
    else:
        er = 0
    # Smoothing constants
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1) # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Calculate KAMA
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc * (close_1d[i] - kama[i-1])
    
    # Align KAMA to 1d timeframe (already aligned as we're using 1d data)
    kama_aligned = kama  # Already on 1d timeframe
    
    # Calculate RSI on 1d data
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    # Prepend first value to match length
    rsi = np.concatenate([[50], rsi])
    
    # Align RSI to 1d timeframe
    rsi_aligned = rsi  # Already on 1d timeframe
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=50, min_periods=50).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate Choppiness Index on 1d data
    atr_period = 14
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[0], tr])  # Align with close
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Calculate highest high and lowest low over period
    hh = pd.Series(high).rolling(window=atr_period, min_periods=atr_period).max().values
    ll = pd.Series(low).rolling(window=atr_period, min_periods=atr_period).min().values
    
    # Choppiness Index: 100 * log10(sum(TR) / (HH - LL)) / log10(period)
    sum_tr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).sum().values
    choppiness = 100 * np.log10(sum_tr / (hh - ll + 1e-10)) / np.log10(atr_period)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(ema_1w_aligned[i]) or np.isnan(choppiness[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above KAMA AND RSI > 50 AND chop < 61.8 (trending) AND weekly uptrend
            if close[i] > kama_aligned[i] and rsi_aligned[i] > 50 and choppiness[i] < 61.8 and close[i] > ema_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA AND RSI < 50 AND chop < 61.8 (trending) AND weekly downtrend
            elif close[i] < kama_aligned[i] and rsi_aligned[i] < 50 and choppiness[i] < 61.8 and close[i] < ema_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price falls below KAMA OR RSI < 40 OR chop > 61.8 (ranging) OR weekly downtrend
            if close[i] < kama_aligned[i] or rsi_aligned[i] < 40 or choppiness[i] > 61.8 or close[i] < ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price rises above KAMA OR RSI > 60 OR chop > 61.8 (ranging) OR weekly uptrend
            if close[i] > kama_aligned[i] or rsi_aligned[i] > 60 or choppiness[i] > 61.8 or close[i] > ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals