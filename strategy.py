#!/usr/bin/env python3
"""
4h_1d_Camarilla_Pivot_Breakout_With_Volume_Confirmation
Hypothesis: Daily Camarilla pivot levels provide tighter support/resistance zones than classic pivots.
Breakouts above H4 or below L4 with volume expansion indicate strong momentum. The 4h EMA50
filter ensures trades align with intermediate-term trend, reducing whipsaws in choppy markets.
Volume confirmation filters out false breakouts. Works in bull markets (breakouts continue)
and bear markets (fades at resistance via trend filter). Targets 25-40 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for each daily bar
    # H4 = Close + 1.5 * (High - Low)
    # L4 = Close - 1.5 * (High - Low)
    camarilla_h4 = close_1d + 1.5 * (high_1d - low_1d)
    camarilla_l4 = close_1d - 1.5 * (high_1d - low_1d)
    # Optional: H3/L3 for context
    camarilla_h3 = close_1d + 1.25 * (high_1d - low_1d)
    camarilla_l3 = close_1d - 1.25 * (high_1d - low_1d)
    
    # Align daily Camarilla levels to 4h
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Volume confirmation: current volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume > (vol_ma_20 * 1.8)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(100, n):
        # Skip if any required data is not ready
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. Breakout above H4 with volume expansion
        # 2. Must be above 4h EMA50 for trend alignment
        breakout_long = (close[i] > camarilla_h4_aligned[i]) and volume_expansion[i]
        long_condition = breakout_long and (close[i] > ema_50_aligned[i])
        
        # Short conditions:
        # 1. Breakdown below L4 with volume expansion
        # 2. Must be below 4h EMA50 for trend alignment
        breakdown_short = (close[i] < camarilla_l4_aligned[i]) and volume_expansion[i]
        short_condition = breakdown_short and (close[i] < ema_50_aligned[i])
        
        if long_condition and position != 1:
            position = 1
            signals[i] = position_size
        elif short_condition and position != -1:
            position = -1
            signals[i] = -position_size
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "4h_1d_Camarilla_Pivot_Breakout_With_Volume_Confirmation"
timeframe = "4h"
leverage = 1.0