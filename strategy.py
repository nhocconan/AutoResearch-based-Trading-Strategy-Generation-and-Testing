#!/usr/bin/env python3
"""
12h_Price_Action_Reversal_Strategy_V1
Hypothesis: Trade reversals at 12h extremes using 1-week pivot points with volume confirmation.
Long when price touches weekly S1 pivot with bullish rejection candle and volume spike.
Short when price touches weekly R1 pivot with bearish rejection candle and volume spike.
Uses weekly pivot levels as institutional support/resistance, reducing false breakouts.
Volume spike (>1.5x 24-period average) confirms institutional interest.
Works in bull/bear: mean reversion at extremes works in ranging markets, trend continuation in strong moves.
Target: 80-120 total trades over 4 years (20-30/year) with position size 0.25.
"""

name = "12h_Price_Action_Reversal_Strategy_V1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Get 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points (standard formula)
    # Pivot = (H + L + C)/3
    # R1 = 2*P - L
    # S1 = 2*P - H
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    r1_1w = 2 * pivot_1w - low_1w
    s1_1w = 2 * pivot_1w - high_1w
    
    # Align weekly pivots to 12h timeframe (wait for weekly close)
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # Calculate volume spike (>1.5x 24-period average for confirmation)
    vol_ma24 = np.full_like(volume, np.nan)
    for i in range(24, len(volume)):
        vol_ma24[i] = np.mean(volume[i-24:i])
    volume_spike = volume > (1.5 * vol_ma24)
    
    # Calculate candlestick patterns for rejection
    # Bullish rejection: close > open and close > previous close (hammer-like)
    bullish_rejection = (close > open_price) & (close > np.roll(close, 1))
    # Bearish rejection: close < open and close < previous close (shooting star-like)
    bearish_rejection = (close < open_price) & (close < np.roll(close, 1))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_1w_aligned[i]) or np.isnan(r1_1w_aligned[i]) or 
            np.isnan(s1_1w_aligned[i]) or np.isnan(close[i]) or 
            np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price touches S1 with bullish rejection and volume spike
            if (low[i] <= s1_1w_aligned[i] * 1.001) and bullish_rejection[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price touches R1 with bearish rejection and volume spike
            elif (high[i] >= r1_1w_aligned[i] * 0.999) and bearish_rejection[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price reaches pivot or shows bearish rejection at R1
            if (close[i] >= pivot_1w_aligned[i] * 0.999) or \
               (high[i] >= r1_1w_aligned[i] * 0.999 and bearish_rejection[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price reaches pivot or shows bullish rejection at S1
            if (close[i] <= pivot_1w_aligned[i] * 1.001) or \
               (low[i] <= s1_1w_aligned[i] * 1.001 and bullish_rejection[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals