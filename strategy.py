#!/usr/bin/env python3
"""
12h_1d_Momentum_Camarilla_With_Volume
Hypothesis: 12-hour breakout of daily Camarilla H4/L4 levels with volume confirmation (>1.8x 30-period average) and momentum filter (price > 50-period SMA). Uses discrete position sizing (0.25) to minimize churn. Designed for low-frequency, high-probability trades in both bull and bear markets. Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_Momentum_Camarilla_With_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
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
    
    # Calculate Camarilla H4/L4 levels for each day
    H4 = np.zeros(len(df_1d))
    L4 = np.zeros(len(df_1d))
    
    for i in range(len(df_1d)):
        range_ = high_1d[i] - low_1d[i]
        if range_ <= 0:
            H4[i] = L4[i] = close_1d[i]
        else:
            H4[i] = close_1d[i] + range_ * 1.1 / 2
            L4[i] = close_1d[i] - range_ * 1.1 / 2
    
    # Align Camarilla levels to 12h timeframe
    H4_12h = align_htf_to_ltf(prices, df_1d, H4)
    L4_12h = align_htf_to_ltf(prices, df_1d, L4)
    
    # === MOMENTUM FILTER (50-period SMA) ===
    sma_50 = pd.Series(close).rolling(window=50, min_periods=50).mean().values
    
    # === VOLUME FILTER ===
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if not ready
        if (np.isnan(H4_12h[i]) or np.isnan(L4_12h[i]) or 
            np.isnan(sma_50[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Breakout conditions
        # Long: Price breaks above H4 with volume + price above SMA50
        long_breakout = (close[i] > H4_12h[i]) and (vol_ratio[i] > 1.8) and (close[i] > sma_50[i])
        
        # Short: Price breaks below L4 with volume + price below SMA50
        short_breakout = (close[i] < L4_12h[i]) and (vol_ratio[i] > 1.8) and (close[i] < sma_50[i])
        
        # Exit: Price returns to opposite H3/L3 level
        # Calculate H3/L3 for exit
        H3 = np.zeros(len(df_1d))
        L3 = np.zeros(len(df_1d))
        for i_1d in range(len(df_1d)):
            range_ = high_1d[i_1d] - low_1d[i_1d]
            if range_ <= 0:
                H3[i_1d] = L3[i_1d] = close_1d[i_1d]
            else:
                H3[i_1d] = close_1d[i_1d] + range_ * 1.1 / 4
                L3[i_1d] = close_1d[i_1d] - range_ * 1.1 / 4
        H3_12h = align_htf_to_ltf(prices, df_1d, H3)
        L3_12h = align_htf_to_ltf(prices, df_1d, L3)
        
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