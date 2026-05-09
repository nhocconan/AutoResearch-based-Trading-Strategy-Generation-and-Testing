#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Choppiness_Keltner_Breakout"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ATR, EMA, and Choppiness
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # True Range components
    prev_close = df_1d['close'].shift(1).values
    tr1 = df_1d['high'].values - df_1d['low'].values
    tr2 = np.abs(df_1d['high'].values - prev_close)
    tr3 = np.abs(df_1d['low'].values - prev_close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14)
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # EMA(20) for Keltner mid
    ema20 = pd.Series(df_1d['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Choppiness Index (14)
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    hh = df_1d['high'].rolling(window=14, min_periods=14).max().values
    ll = df_1d['low'].rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (hh - ll)) / np.log10(14)
    
    # Keltner Bands
    upper = ema20 + 2.0 * atr
    lower = ema20 - 2.0 * atr
    
    # Align to 4h
    upper_4h = align_htf_to_ltf(prices, df_1d, upper)
    lower_4h = align_htf_to_ltf(prices, df_1d, lower)
    chop_4h = align_htf_to_ltf(prices, df_1d, chop)
    ema20_4h = align_htf_to_ltf(prices, df_1d, ema20)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 14  # Need enough for ATR and Choppiness
    
    for i in range(start_idx, n):
        if (np.isnan(upper_4h[i]) or np.isnan(lower_4h[i]) or
            np.isnan(chop_4h[i]) or np.isnan(ema20_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        up = upper_4h[i]
        low = lower_4h[i]
        chop_val = chop_4h[i]
        mid = ema20_4h[i]
        
        if position == 0:
            # Enter long: price above upper Keltner in low chop (trending)
            if close[i] > up and chop_val < 38.2:
                signals[i] = 0.25
                position = 1
            # Enter short: price below lower Keltner in low chop (trending)
            elif close[i] < low and chop_val < 38.2:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price below EMA (trend end) or high chop (range)
            if close[i] < mid or chop_val > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price above EMA (trend end) or high chop (range)
            if close[i] > mid or chop_val > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals