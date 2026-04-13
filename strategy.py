#!/usr/bin/env python3
"""
12h_1w_1d_Camarilla_Pivot_Volume_Strategy
Hypothesis: Trade Camarilla pivot level touches on 12h timeframe with volume confirmation and 1d trend filter.
Camarilla levels (H3, L3) act as intraday support/resistance. Price approaching these levels with
volume expansion indicates institutional interest. 1d EMA200 ensures trades align with long-term trend.
Works in both bull (buy dips to L3 in uptrend) and bear (sell rallies to H3 in downtrend) markets.
Target: 15-25 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    close_prev = df_1d['close'].shift(1).values
    
    # Camarilla formulas
    range_prev = high_prev - low_prev
    H3 = close_prev + (range_prev * 1.1 / 2)
    L3 = close_prev - (range_prev * 1.1 / 2)
    H4 = close_prev + (range_prev * 1.1)
    L4 = close_prev - (range_prev * 1.1)
    
    # Align Camarilla levels to 12h
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    
    # Get weekly data for trend filter (1-week EMA200 equivalent)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Weekly EMA20 (approximates daily EMA200 on 12h chart)
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Volume confirmation: current volume > 1.8x 30-period average
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean()
    volume_expansion = volume > (vol_ma_30 * 1.8)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(200, n):
        # Skip if any required data is not ready
        if (np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or 
            np.isnan(H4_aligned[i]) or np.isnan(L4_aligned[i]) or
            np.isnan(ema_20_aligned[i]) or np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            continue
        
        # Long: price approaches L3 from below with volume expansion and above weekly EMA20
        long_condition = (
            (low[i] <= L3_aligned[i] * 1.001) and  # Allow small tolerance for touch
            (close[i] > L3_aligned[i]) and         # Confirmed bounce
            volume_expansion[i] and
            (close[i] > ema_20_aligned[i])
        )
        
        # Short: price approaches H3 from above with volume expansion and below weekly EMA20
        short_condition = (
            (high[i] >= H3_aligned[i] * 0.999) and # Allow small tolerance for touch
            (close[i] < H3_aligned[i]) and         # Confirmed rejection
            volume_expansion[i] and
            (close[i] < ema_20_aligned[i])
        )
        
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

name = "12h_1w_1d_Camarilla_Pivot_Volume_Strategy"
timeframe = "12h"
leverage = 1.0