#!/usr/bin/env python3
"""
12h_KAMA_Trend_With_1d_Trend_Filter
Hypothesis: 12-hour KAMA trend with 1-day trend filter and volume confirmation
captures medium-term trends while avoiding whipsaws. KAMA adapts to market
efficiency, providing better trend signals than simple moving averages.
The 1-day EMA filter ensures alignment with higher timeframe trend, reducing
counter-trend trades. Volume confirmation ensures breakouts have conviction.
Works in both bull and bear markets by following the dominant trend.
"""

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
    
    # Get 1-day data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1-day OHLC arrays
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1-day EMA trend filter (34-period)
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 12-hour KAMA trend (ER=10, slow=2, fast=30)
    # Efficiency Ratio = |change| / sum(|changes|)
    change = np.abs(np.diff(close, prepend=close[0]))
    abs_change = np.abs(np.diff(close, prepend=close[0]))
    sum_abs_change = pd.Series(abs_change).rolling(window=10, min_periods=10).sum().values
    er = np.divide(change, sum_abs_change, out=np.zeros_like(change), where=sum_abs_change!=0)
    # Smoothing constant = [ER * (fastest - slowest) + slowest]^2
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # 12-hour volume filter: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Warmup for KAMA and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(kama[i]) or 
            np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        kama_val = kama[i]
        ema_trend = ema_1d_aligned[i]
        vol_ok = volume_filter[i]
        
        if position == 0:
            # Long: price above KAMA and above 1-day EMA with volume
            if price > kama_val and price > ema_trend and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA and below 1-day EMA with volume
            elif price < kama_val and price < ema_trend and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long if price crosses below KAMA or trend reverses
            if price < kama_val or price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short if price crosses above KAMA or trend reverses
            if price > kama_val or price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_KAMA_Trend_With_1d_Trend_Filter"
timeframe = "12h"
leverage = 1.0