#!/usr/bin/env python3
"""
Hypothesis: 4-hour strategy using 12-hour Camarilla pivot levels with volume confirmation and momentum filter.
Long when price crosses above H4 pivot level with volume surge and positive momentum.
Short when price crosses below L4 pivot level with volume surge and negative momentum.
Exit when price returns to central pivot (P) or momentum reverses.
Designed for low turnover: ~20-30 trades/year per symbol to minimize fee drag.
Works in bull markets via breakout longs and in bear via short-side symmetry.
Uses proven Camarilla pivot structure from winning strategies.
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
    
    # Load 12-hour data once for Camarilla pivot calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12-hour Camarilla levels (based on previous day's range)
    # P = (H + L + C) / 3
    # H4 = P + 1.1 * (H - L) / 2
    # L4 = P - 1.1 * (H - L) / 2
    typical_price = (high_12h + low_12h + close_12h) / 3
    range_hl = high_12h - low_12h
    
    P = typical_price
    H4 = P + 1.1 * range_hl / 2
    L4 = P - 1.1 * range_hl / 2
    
    # Momentum filter: 12-period ROC
    roc_period = 12
    roc = np.zeros_like(close_12h)
    for i in range(roc_period, len(close_12h)):
        if close_12h[i - roc_period] != 0:
            roc[i] = (close_12h[i] - close_12h[i - roc_period]) / close_12h[i - roc_period] * 100
    
    # Volume filter: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(30, n):
        # 12-hour index (2 bars per day for 4h timeframe)
        idx_12h = i // 2
        if idx_12h < 20:  # need enough for calculations
            continue
        
        # Use previous 12h values to avoid look-ahead (previous completed bar)
        prev_idx = idx_12h - 1
        if prev_idx < 0:
            continue
            
        # Get Camarilla levels from previous 12h bar
        P_prev = P[prev_idx] if prev_idx < len(P) else P[-1]
        H4_prev = H4[prev_idx] if prev_idx < len(H4) else H4[-1]
        L4_prev = L4[prev_idx] if prev_idx < len(L4) else L4[-1]
        roc_prev = roc[prev_idx] if prev_idx < len(roc) else roc[-1]
        
        if np.isnan(P_prev) or np.isnan(H4_prev) or np.isnan(L4_prev) or np.isnan(roc_prev):
            continue
        
        # Create arrays for alignment (using previous values)
        P_arr = np.full(len(df_12h), P_prev)
        H4_arr = np.full(len(df_12h), H4_prev)
        L4_arr = np.full(len(df_12h), L4_prev)
        roc_arr = np.full(len(df_12h), roc_prev)
        
        P_4h = align_htf_to_ltf(prices, df_12h, P_arr)[i]
        H4_4h = align_htf_to_ltf(prices, df_12h, H4_arr)[i]
        L4_4h = align_htf_to_ltf(prices, df_12h, L4_arr)[i]
        roc_4h = align_htf_to_ltf(prices, df_12h, roc_arr)[i]
        
        if position == 0:
            # Long: price crosses above H4 + volume surge + positive momentum
            if (close[i] > H4_4h and 
                volume[i] > vol_ma[i] * 1.5 and
                roc_4h > 0):
                position = 1
                signals[i] = position_size
            # Short: price crosses below L4 + volume surge + negative momentum
            elif (close[i] < L4_4h and 
                  volume[i] > vol_ma[i] * 1.5 and
                  roc_4h < 0):
                position = -1
                signals[i] = -position_size
        elif position == 1:
            # Exit: price returns to central pivot or momentum turns negative
            if close[i] < P_4h or roc_4h < 0:
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Exit: price returns to central pivot or momentum turns positive
            if close[i] > P_4h or roc_4h > 0:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "4h_12h_Camarilla_Volume_Momentum"
timeframe = "4h"
leverage = 1.0