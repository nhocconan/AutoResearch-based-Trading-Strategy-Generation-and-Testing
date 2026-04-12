#!/usr/bin/env python3
"""
1h_4h_1d_Camarilla_Pullback_v1
Hypothesis: On 1h timeframe, buy pullbacks to Camarilla L3 in 4h uptrend (price > 1d EMA200) and sell rallies to H3 in 4h downtrend (price < 1d EMA200).
Trades only during active session (08-20 UTC) to reduce noise. Uses 1h for precise entry timing, 4h/1d for trend and structure.
Target: 60-150 trades over 4 years (15-37/year) with low turnover to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_Camarilla_Pullback_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # --- 4h data for Camarilla levels and trend ---
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Camarilla levels using prior 4h bar
    range_4h = high_4h - low_4h
    close_prev = np.concatenate([[close_4h[0]], close_4h[:-1]])  # shift(1)
    
    h3 = close_prev + (range_4h * 1.1 / 4)
    l3 = close_prev - (range_4h * 1.1 / 4)
    
    # --- 1d data for trend filter (EMA200) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # EMA200 with proper initialization
    ema_200 = np.full_like(close_1d, np.nan)
    alpha = 2 / (200 + 1)
    for i in range(len(close_1d)):
        if i == 0:
            ema_200[i] = close_1d[i]
        elif np.isnan(ema_200[i-1]):
            ema_200[i] = close_1d[i]
        else:
            ema_200[i] = alpha * close_1d[i] + (1 - alpha) * ema_200[i-1]
    
    # --- Align HTF data to 1h ---
    h3_aligned = align_htf_to_ltf(prices, df_4h, h3)
    l3_aligned = align_htf_to_ltf(prices, df_4h, l3)
    ema200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    
    # --- Session filter: 08-20 UTC ---
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):
        # Skip if not in session or data not ready
        if not in_session[i] or np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or np.isnan(ema200_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Trend: 4h uptrend = price > 1d EMA200, downtrend = price < 1d EMA200
        uptrend = close[i] > ema200_aligned[i]
        downtrend = close[i] < ema200_aligned[i]
        
        # Pullback entries: buy near L3 in uptrend, sell near H3 in downtrend
        long_setup = uptrend and (low[i] <= l3_aligned[i]) and (close[i] > l3_aligned[i])
        short_setup = downtrend and (high[i] >= h3_aligned[i]) and (close[i] < h3_aligned[i])
        
        # Mean reversion exit: reverse at opposite Camarilla level
        exit_long = downtrend and (high[i] >= h3_aligned[i])
        exit_short = uptrend and (low[i] <= l3_aligned[i])
        
        if long_setup and position != 1:
            position = 1
            signals[i] = 0.20
        elif short_setup and position != -1:
            position = -1
            signals[i] = -0.20
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals