#!/usr/bin/env python3
"""
4h_1d_1w_Camarilla_Breakout_TripleFilter
Hypothesis: Daily Camarilla S4/R4 levels act as institutional support/resistance. 
Breakout confirmed by volume surge and weekly trend filter (EMA50) captures momentum with low false signals.
Designed for low trade frequency (15-25/year) to avoid fee drag while capturing strong trending moves in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_1w_Camarilla_Breakout_TripleFilter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === DAILY CAMARILLA LEVELS ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla calculations
    pivot = (high_1d + low_1d + close_1d) / 3.0
    rang = high_1d - low_1d
    r4 = close_1d + rang * 1.1 / 2.0  # Resistance level 4
    s4 = close_1d - rang * 1.1 / 2.0  # Support level 4
    
    # Align to 4h timeframe (wait for daily close)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    
    # === WEEKLY TREND FILTER ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # === VOLUME CONFIRMATION (4h) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):
        # Skip if not ready
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(pivot_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # LONG: Breakout above R4 with volume surge and weekly uptrend
        long_signal = (close[i] > r4_aligned[i]) and (vol_ratio[i] > 2.0) and (close[i] > ema50_1w_aligned[i])
        
        # SHORT: Breakdown below S4 with volume surge and weekly downtrend
        short_signal = (close[i] < s4_aligned[i]) and (vol_ratio[i] > 2.0) and (close[i] < ema50_1w_aligned[i])
        
        # EXIT: Return to pivot area (mean reversion)
        exit_long = (position == 1) and (close[i] < pivot_aligned[i])
        exit_short = (position == -1) and (close[i] > pivot_aligned[i])
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals