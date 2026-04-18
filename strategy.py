#!/usr/bin/env python3
"""
6h_Weekly_Range_Bound_Strategy
Hypothesis: In ranging markets (common in 2025), price oscillates between weekly support/resistance. 
Go long near weekly S1/S2 with bullish momentum, short near weekly R1/R2 with bearish momentum.
Uses weekly pivot points for structure and 6-hour RSI for timing. Works in both bull and bear by 
fading extremes in ranging conditions. Targets 10-20 trades/year with position size 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for pivot points
    df_w = get_htf_data(prices, '1w')
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    close_w = df_w['close'].values
    
    # Calculate weekly pivot points (using prior week's data)
    # Pivot = (H + L + C)/3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    pivot_w = (high_w + low_w + close_w) / 3.0
    r1_w = 2 * pivot_w - low_w
    s1_w = 2 * pivot_w - high_w
    r2_w = pivot_w + (high_w - low_w)
    s2_w = pivot_w - (high_w - low_w)
    
    # Align weekly levels to 6h timeframe (wait for weekly bar close)
    r1_6h = align_htf_to_ltf(prices, df_w, r1_w)
    s1_6h = align_htf_to_ltf(prices, df_w, s1_w)
    r2_6h = align_htf_to_ltf(prices, df_w, r2_w)
    s2_6h = align_htf_to_ltf(prices, df_w, s2_w)
    
    # Calculate 6-hour RSI for momentum confirmation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    for i in range(14, n):
        if i == 14:
            avg_gain[i] = np.mean(gain[1:15])
            avg_loss[i] = np.mean(loss[1:15])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 14  # need RSI
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_6h[i]) or np.isnan(s1_6h[i]) or 
            np.isnan(r2_6h[i]) or np.isnan(s2_6h[i]) or
            np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: near weekly S1/S2 with bullish momentum (RSI > 50)
            near_support = (close[i] <= s1_6h[i] * 1.02 or close[i] <= s2_6h[i] * 1.02)
            bullish_momentum = rsi[i] > 50
            
            if near_support and bullish_momentum:
                signals[i] = 0.25
                position = 1
            # Short entry: near weekly R1/R2 with bearish momentum (RSI < 50)
            elif (close[i] >= r1_6h[i] * 0.98 or close[i] >= r2_6h[i] * 0.98) and rsi[i] < 50:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: RSI becomes overbought or price reaches resistance
            if rsi[i] > 70 or close[i] >= r1_6h[i] * 0.98:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI becomes oversold or price reaches support
            if rsi[i] < 30 or close[i] <= s1_6h[i] * 1.02:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Weekly_Range_Bound_Strategy"
timeframe = "6h"
leverage = 1.0