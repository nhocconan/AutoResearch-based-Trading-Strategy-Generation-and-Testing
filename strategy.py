#!/usr/bin/env python3
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
    
    # Get 1d data for Williams Vix Fix
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 22:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams Vix Fix (WVF) on daily data
    # WVF = ((Highest Close in n-period - Low) / Highest Close in n-period) * 100
    n_period = 22
    highest_close = np.full(len(close_1d), np.nan)
    for i in range(n_period - 1, len(close_1d)):
        highest_close[i] = np.max(close_1d[i - n_period + 1:i + 1])
    
    wvf = np.full(len(close_1d), np.nan)
    for i in range(n_period - 1, len(close_1d)):
        if highest_close[i] != 0:
            wvf[i] = ((highest_close[i] - low_1d[i]) / highest_close[i]) * 100
    
    # Calculate EMA of WVF for signal smoothing
    wvf_ema = np.full(len(wvf), np.nan)
    ema_period = 9
    for i in range(len(wvf)):
        if i == ema_period - 1:
            wvf_ema[i] = np.mean(wvf[ema_period - 1:i + 1])
        elif i >= ema_period:
            wvf_ema[i] = (wvf[i] * (2 / (ema_period + 1)) + 
                          wvf_ema[i - 1] * (1 - (2 / (ema_period + 1))))
    
    # Align WVF and its EMA to 6h timeframe
    wvf_aligned = align_htf_to_ltf(prices, df_1d, wvf)
    wvf_ema_aligned = align_htf_to_ltf(prices, df_1d, wvf_ema)
    
    # Get 12h data for trend filter (EMA34)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA34
    ema_12h = np.full(len(close_12h), np.nan)
    ema_period_12h = 34
    for i in range(len(close_12h)):
        if i == ema_period_12h - 1:
            ema_12h[i] = np.mean(close_12h[ema_period_12h - 1:i + 1])
        elif i >= ema_period_12h:
            ema_12h[i] = (close_12h[i] * (2 / (ema_period_12h + 1)) + 
                          ema_12h[i - 1] * (1 - (2 / (ema_period_12h + 1))))
    
    # Align 12h EMA34 to 6h timeframe
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need WVF, WVF EMA, and 12h EMA
    start_idx = max(n_period - 1, ema_period - 1, ema_period_12h - 1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(wvf_aligned[i]) or np.isnan(wvf_ema_aligned[i]) or 
            np.isnan(ema_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        wvf_val = wvf_aligned[i]
        wvf_ema_val = wvf_ema_aligned[i]
        ema_trend = ema_12h_aligned[i]
        price = close[i]
        
        if position == 0:
            # Long: WVF below EMA (fear spike) and price above 12h EMA (uptrend)
            if wvf_val < wvf_ema_val and price > ema_trend:
                signals[i] = size
                position = 1
            # Short: WVF above EMA (complacency) and price below 12h EMA (downtrend)
            elif wvf_val > wvf_ema_val and price < ema_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: WVF crosses above EMA (fear subsiding) or trend fails
            if wvf_val > wvf_ema_val or price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: WVF crosses below EMA (complacency ending) or trend fails
            if wvf_val < wvf_ema_val or price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6H_Williams_Vix_Fix_Reversal"
timeframe = "6h"
leverage = 1.0