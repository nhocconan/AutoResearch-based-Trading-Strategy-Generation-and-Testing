#!/usr/bin/env python3
"""
Hypothesis: 6h strategy using KAMA (Kaufman Adaptive Moving Average) with 1-week trend filter and volume confirmation.
- KAMA adapts to market noise: faster in trends, slower in ranges, reducing whipsaw.
- Long when price > KAMA(10) and price > 1-week EMA50 with volume > 1.5x 20-volume MA.
- Short when price < KAMA(10) and price < 1-week EMA50 with volume > 1.5x 20-volume MA.
- Exit when price crosses back across KAMA(10).
- Uses 1-week trend filter to avoid counter-trend trades in bear markets.
- Designed for 6h timeframe with strict entry conditions to limit trades to 50-150 total over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def kama(close, length=10, fast=2, slow=30):
    """Kaufman Adaptive Moving Average"""
    change = np.abs(np.diff(close, n=length))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
    kama = np.full_like(close, np.nan, dtype=float)
    kama[length] = close[length]
    for i in range(length+1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    return kama

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-week data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1-week EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate KAMA(10) on 6h data
    kama_val = kama(close, length=10, fast=2, slow=30)
    
    # Volume confirmation: 20-volume MA
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 30  # warmup for KAMA and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(kama_val[i]) or 
            np.isnan(volume_ma_20.iloc[i]) or
            np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20.iloc[i]
        kama = kama_val[i]
        ema_1w = ema_50_1w_aligned[i]
        
        if position == 0:
            # Look for signals with volume confirmation and weekly trend filter
            # Long: price > KAMA, price > weekly EMA50, volume spike
            if price > kama and price > ema_1w and vol > 1.5 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Short: price < KAMA, price < weekly EMA50, volume spike
            elif price < kama and price < ema_1w and vol > 1.5 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit when price crosses below KAMA
            if price < kama:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when price crosses above KAMA
            if price > kama:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_KAMA_Volume_1wEMA50"
timeframe = "6h"
leverage = 1.0