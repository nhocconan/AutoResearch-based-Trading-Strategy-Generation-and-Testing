#!/usr/bin/env python3
"""
12h_1w_1d_Camarilla_Breakout_Volume_Strategy
Hypothesis: Camarilla pivot levels (resistance/support) from daily timeframe identify key institutional levels.
Breakout above R4 or below S4 with volume confirmation and weekly trend filter captures institutional flow.
Works in both bull (breakouts above R4) and bear (breakdowns below S4) markets.
Low trade frequency (~20-30/year) minimizes fee drag while capturing significant moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_1d_Camarilla_Breakout_Volume_Strategy"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === DAILY DATA FOR CAMARILLA PIVOTS ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    # Pivot = (High + Low + Close) / 3
    # Range = High - Low
    # R4 = Close + Range * 1.1/2
    # S4 = Close - Range * 1.1/2
    pivot = (high_1d + low_1d + close_1d) / 3.0
    rang = high_1d - low_1d
    r4 = close_1d + rang * 1.1 / 2.0
    s4 = close_1d - rang * 1.1 / 2.0
    
    # Align Camarilla levels to 12h timeframe (wait for daily bar to close)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # === WEEKLY TREND FILTER ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # 50-period EMA for weekly trend
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # === VOLUME CONFIRMATION (12h) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if not ready
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Breakout above R4 with volume confirmation and weekly uptrend
        long_breakout = (close[i] > r4_aligned[i]) and (vol_ratio[i] > 1.5) and (close[i] > ema50_1w_aligned[i])
        
        # Breakdown below S4 with volume confirmation and weekly downtrend
        short_breakdown = (close[i] < s4_aligned[i]) and (vol_ratio[i] > 1.5) and (close[i] < ema50_1w_aligned[i])
        
        # Exit when price returns to pivot area (mean reversion)
        pivot_align = align_htf_to_ltf(prices, df_1d, pivot)
        exit_long = close[i] < pivot_align[i] and position == 1
        exit_short = close[i] > pivot_align[i] and position == -1
        
        # Execute trades
        if long_breakout and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_breakdown and position != -1:
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