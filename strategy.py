#!/usr/bin/env python3
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
    volume = prices['volume'].values
    
    # Get weekly data for trend (called ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA(20) for trend
    close_1w = df_1w['close'].values
    ema_20_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 20:
        multiplier = 2 / (20 + 1)
        ema_20_1w[19] = np.mean(close_1w[:20])
        for i in range(20, len(close_1w)):
            ema_20_1w[i] = (close_1w[i] * multiplier) + (ema_20_1w[i-1] * (1 - multiplier))
    
    # Align weekly EMA to daily
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate daily ATR(14) for volatility filter
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = np.full(n, np.nan)
    if n >= 14:
        atr[13] = np.mean(tr[:14])
        for i in range(14, n):
            atr[i] = (tr[i] * 0.1) + (atr[i-1] * 0.9)  # Wilder's smoothing
    
    # Calculate daily Donchian Channel(20)
    upper_dc = np.full(n, np.nan)
    lower_dc = np.full(n, np.nan)
    if n >= 20:
        for i in range(19, n):
            upper_dc[i] = np.max(high[i-19:i+1])
            lower_dc[i] = np.min(low[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25
    
    # Warmup period
    start_idx = max(20, 14) + 5
    
    for i in range(start_idx, n):
        if (np.isnan(ema_20_1w_aligned[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(upper_dc[i]) or 
            np.isnan(lower_dc[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: Price above weekly EMA and breaks above upper Donchian with low volatility
            if price > ema_20_1w_aligned[i] and price > upper_dc[i] and atr[i] < np.nanmedian(atr[max(0,i-20):i]):
                signals[i] = size
                position = 1
            # Short: Price below weekly EMA and breaks below lower Donchian with low volatility
            elif price < ema_20_1w_aligned[i] and price < lower_dc[i] and atr[i] < np.nanmedian(atr[max(0,i-20):i]):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price crosses below weekly EMA or breaks below lower Donchian
            if price < ema_20_1w_aligned[i] or price < lower_dc[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Price crosses above weekly EMA or breaks above upper Donchian
            if price > ema_20_1w_aligned[i] or price > upper_dc[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_WeeklyEMA20_Donchian20_LowVol"
timeframe = "1d"
leverage = 1.0