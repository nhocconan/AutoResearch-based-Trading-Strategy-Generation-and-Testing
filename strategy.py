#!/usr/bin/env python3
"""
1h_4h1d_RSI_Pivot_Confluence_v1
Concept: RSI mean reversion with 4h/1d trend filter and pivot confluence.
- Long: RSI(14) < 30 AND 4h EMA(21) > EMA(55) AND 1d EMA(55) > EMA(89) AND price > weekly pivot S1
- Short: RSI(14) > 70 AND 4h EMA(21) < EMA(55) AND 1d EMA(55) < EMA(89) AND price < weekly pivot R1
- Exit: RSI crosses back to neutral (40-60) or opposite extreme
- Session filter: 08-20 UTC only
- Position sizing: 0.20
- Target: 80-120 total trades over 4 years (20-30/year)
- Works in bull/bear: Uses higher timeframe trend alignment to avoid counter-trend trades
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h1d_RSI_Pivot_Confluence_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Get 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Get weekly data for pivots ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # === 4h: EMA Trend Filter ===
    close_4h = df_4h['close'].values
    ema21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema55_4h = pd.Series(close_4h).ewm(span=55, adjust=False, min_periods=55).mean().values
    ema21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema21_4h)
    ema55_4h_aligned = align_htf_to_ltf(prices, df_4h, ema55_4h)
    
    # === 1d: EMA Trend Filter ===
    close_1d = df_1d['close'].values
    ema55_1d = pd.Series(close_1d).ewm(span=55, adjust=False, min_periods=55).mean().values
    ema89_1d = pd.Series(close_1d).ewm(span=89, adjust=False, min_periods=89).mean().values
    ema55_1d_aligned = align_htf_to_ltf(prices, df_1d, ema55_1d)
    ema89_1d_aligned = align_htf_to_ltf(prices, df_1d, ema89_1d)
    
    # === Weekly: Pivot Points (using prior week) ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate pivots based on prior week
    pivot = (high_1w + low_1w + close_1w) / 3
    r1 = 2 * pivot - low_1w
    s1 = 2 * pivot - high_1w
    
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # === 1h: RSI ===
    close = prices['close'].values
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / np.where(avg_loss == 0, 1e-10, avg_loss)
    rsi = 100 - (100 / (1 + rs))
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure enough data
    
    for i in range(start_idx, n):
        # Session check
        if not (8 <= hours[i] <= 20):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get values
        rsi_val = rsi[i]
        ema21_4h_val = ema21_4h_aligned[i]
        ema55_4h_val = ema55_4h_aligned[i]
        ema55_1d_val = ema55_1d_aligned[i]
        ema89_1d_val = ema89_1d_aligned[i]
        pivot_val = pivot_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(rsi_val) or np.isnan(ema21_4h_val) or np.isnan(ema55_4h_val) or
            np.isnan(ema55_1d_val) or np.isnan(ema89_1d_val) or np.isnan(pivot_val) or
            np.isnan(r1_val) or np.isnan(s1_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI oversold + 4h/1d uptrend + price above S1
            if (rsi_val < 30 and ema21_4h_val > ema55_4h_val and 
                ema55_1d_val > ema89_1d_val and close[i] > s1_val):
                signals[i] = 0.20
                position = 1
            # Short: RSI overbought + 4h/1d downtrend + price below R1
            elif (rsi_val > 70 and ema21_4h_val < ema55_4h_val and 
                  ema55_1d_val < ema89_1d_val and close[i] < r1_val):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: RSI returns to neutral or breaks down
            if rsi_val > 40 or close[i] < s1_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: RSI returns to neutral or breaks up
            if rsi_val < 60 or close[i] > r1_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals