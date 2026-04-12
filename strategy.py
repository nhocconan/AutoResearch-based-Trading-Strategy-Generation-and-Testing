#!/usr/bin/env python3
"""
12h_1d_Camarilla_Breakout_Momentum
Hypothesis: Camarilla pivot levels from daily timeframe act as support/resistance.
Breakouts above resistance (H3/H4) with volume confirmation and 12h trend filter
capture momentum moves. Works in both bull and bear markets by trading breakouts
in direction of 12h trend. Low frequency (~20-30 trades/year) minimizes fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_Camarilla_Breakout_Momentum"
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
    
    # Align Camarilla levels to 12h timeframe
    H3_12h = align_htf_to_ltf(prices, df_1d, H3)
    L3_12h = align_htf_to_ltf(prices, df_1d, L3)
    H4_12h = align_htf_to_ltf(prices, df_1d, H4)
    L4_12h = align_htf_to_ltf(prices, df_1d, L4)
    
    # === 12h TREND FILTER (EMA 50) ===
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # === VOLUME CONFIRMATION ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if not ready
        if (np.isnan(H3_12h[i]) or np.isnan(L3_12h[i]) or 
            np.isnan(H4_12h[i]) or np.isnan(L4_12h[i]) or 
            np.isnan(ema50[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Breakout conditions
        # Long: Price breaks above H3 with volume + above EMA50
        long_breakout = (close[i] > H3_12h[i]) and (vol_ratio[i] > 1.5) and (close[i] > ema50[i])
        
        # Short: Price breaks below L3 with volume + below EMA50
        short_breakout = (close[i] < L3_12h[i]) and (vol_ratio[i] > 1.5) and (close[i] < ema50[i])
        
        # Exit: Price returns to opposite Camarilla level (L3 for long, H3 for short)
        exit_long = (position == 1) and (close[i] < L3_12h[i])
        exit_short = (position == -1) and (close[i] > H3_12h[i])
        
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