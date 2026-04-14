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
    
    # Load weekly data for pivot levels (HH, LL, Close of previous week)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points: P = (H+L+C)/3, R1 = 2*P-L, S1 = 2*P-H
    # We'll use the previous week's values to avoid lookahead
    pivot = np.full_like(close_1w, np.nan)
    r1 = np.full_like(close_1w, np.nan)
    s1 = np.full_like(close_1w, np.nan)
    
    if len(close_1w) >= 2:  # Need at least 2 weeks to have previous week data
        for i in range(1, len(close_1w)):
            # Use previous week's data
            ph = high_1w[i-1]
            pl = low_1w[i-1]
            pc = close_1w[i-1]
            p = (ph + pl + pc) / 3.0
            r = 2 * p - pl
            s = 2 * p - ph
            pivot[i] = p
            r1[i] = r
            s1[i] = s
    
    # Align weekly pivot levels to 6h
    pivot_6h = align_htf_to_ltf(prices, df_1w, pivot)
    r1_6h = align_htf_to_ltf(prices, df_1w, r1)
    s1_6h = align_htf_to_ltf(prices, df_1w, s1)
    
    # Load daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Daily EMA(50) for trend filter
    ema_50_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 50:
        ema_50_1d[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = (close_1d[i] * 2 + ema_50_1d[i-1] * 48) / 50
    
    ema_50_1d_6h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any critical data is NaN
        if (np.isnan(pivot_6h[i]) or 
            np.isnan(r1_6h[i]) or 
            np.isnan(s1_6h[i]) or 
            np.isnan(ema_50_1d_6h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price crosses above R1 with price > daily EMA50
            if close[i] > r1_6h[i] and close[i] > ema_50_1d_6h[i]:
                position = 1
                signals[i] = position_size
            # Short: Price crosses below S1 with price < daily EMA50
            elif close[i] < s1_6h[i] and close[i] < ema_50_1d_6h[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price crosses below pivot OR price < daily EMA50
            if close[i] < pivot_6h[i] or close[i] < ema_50_1d_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price crosses above pivot OR price > daily EMA50
            if close[i] > pivot_6h[i] or close[i] > ema_50_1d_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1w_Pivot_R1S1_DailyEMA50"
timeframe = "6h"
leverage = 1.0