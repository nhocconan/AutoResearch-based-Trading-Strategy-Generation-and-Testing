#!/usr/bin/env python3
"""
12h_KAMA_Trend_With_1d_Trend_Filter
Hypothesis: 12-hour KAMA trend aligned with 1-day EMA34 filter captures sustained momentum while reducing whipsaws.
KAMA adapts to market noise, effective in both trending and ranging markets. The 1-day EMA ensures alignment
with higher timeframe trend. Expects 12-37 trades/year on 12h timeframe with low frequency to minimize fee drag.
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
    volume = prices['volume'].values
    
    # Get 1-day data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1-day EMA34 trend filter
    ema_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # KAMA on 12h timeframe
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # Efficiency ratio
    change = np.abs(np.diff(close_12h, n=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close_12h, n=1)), axis=0)  # 10-period volatility
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    kama = np.zeros_like(close_12h)
    kama[0] = close_12h[0]
    for i in range(1, len(close_12h)):
        kama[i] = kama[i-1] + sc[i] * (close_12h[i] - kama[i-1])
    
    kama_aligned = align_htf_to_ltf(prices, df_12h, kama)
    
    # Volume confirmation: >1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Warmup for KAMA and indicators
    
    for i in range(start_idx, n):
        if (np.isnan(kama_aligned[i]) or np.isnan(ema_1d_aligned[i]) or 
            np.isnan(volume_ok[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        kama_val = kama_aligned[i]
        ema_trend = ema_1d_aligned[i]
        vol_ok = volume_ok[i]
        
        if position == 0:
            # Long: price above KAMA and above 1d EMA with volume
            if price > kama_val and price > ema_trend and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA and below 1d EMA with volume
            elif price < kama_val and price < ema_trend and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long if price crosses below KAMA or trend fails
            if price < kama_val or price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short if price crosses above KAMA or trend fails
            if price > kama_val or price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_KAMA_Trend_With_1d_Trend_Filter"
timeframe = "12h"
leverage = 1.0