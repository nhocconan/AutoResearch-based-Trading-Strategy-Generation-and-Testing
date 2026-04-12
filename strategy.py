#!/usr/bin/env python3
"""
1d_1w_Camarilla_Breakout_With_Trend
Hypothesis: Weekly trend filter + daily Camarilla breakouts with volume confirmation.
In bull markets, buy breakouts above H3/H4 when weekly trend is up.
In bear markets, sell breakdowns below L3/L4 when weekly trend is down.
Weekly trend avoids counter-trend trades, reducing whipsaw. Low frequency (~10-25 trades/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Camarilla_Breakout_With_Trend"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === WEEKLY TREND FILTER (EMA 20) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # === DAILY CAMARILLA PIVOT CALCULATION ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    H3 = np.zeros(len(df_1d))
    L3 = np.zeros(len(df_1d))
    H4 = np.zeros(len(df_1d))
    L4 = np.zeros(len(df_1d))
    
    for i in range(len(df_1d)):
        range_ = high_1d[i] - low_1d[i]
        if range_ <= 0:
            H3[i] = H4[i] = L3[i] = L4[i] = close_1d[i]
        else:
            H3[i] = close_1d[i] + range_ * 1.1 / 4
            L3[i] = close_1d[i] - range_ * 1.1 / 4
            H4[i] = close_1d[i] + range_ * 1.1 / 2
            L4[i] = close_1d[i] - range_ * 1.1 / 2
    
    # Align Camarilla levels to daily timeframe
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    
    # === VOLUME CONFIRMATION ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if not ready
        if (np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or 
            np.isnan(H4_aligned[i]) or np.isnan(L4_aligned[i]) or 
            np.isnan(ema20_1w_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine weekly trend
        weekly_up = close[i] > ema20_1w_aligned[i]
        weekly_down = close[i] < ema20_1w_aligned[i]
        
        # Breakout conditions with weekly trend filter
        # Long: Price breaks above H3 with volume + weekly trend up
        long_breakout = (close[i] > H3_aligned[i]) and (vol_ratio[i] > 1.5) and weekly_up
        
        # Short: Price breaks below L3 with volume + weekly trend down
        short_breakout = (close[i] < L3_aligned[i]) and (vol_ratio[i] > 1.5) and weekly_down
        
        # Exit: Price returns to opposite Camarilla level (L3 for long, H3 for short)
        exit_long = (position == 1) and (close[i] < L3_aligned[i])
        exit_short = (position == -1) and (close[i] > H3_aligned[i])
        
        # Execute trades
        if long_breakout and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_breakout and position != -1:
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