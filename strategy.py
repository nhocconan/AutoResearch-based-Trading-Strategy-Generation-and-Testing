#!/usr/bin/env python3
"""
4h_Angle_of_Descent_v2
Hypothesis: Price action relative to the prior day's high and low, combined with a trend filter,
captures mean-reversion bounces and trend-following continuations. Works in both bull and bear
markets by adapting to price position relative to daily extremes.
Target: 20-50 trades per year on 4h timeframe.
"""

name = "4h_Angle_of_Descent_v2"
timeframe = "4h"
leverage = 1.0

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
    
    # === 1D Data for Daily High and Low ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Align daily high and low to 4h timeframe
    high_1d_aligned = align_htf_to_ltf(prices, df_1d, high_1d)
    low_1d_aligned = align_htf_to_ltf(prices, df_1d, low_1d)
    
    # === 1D EMA for Trend Filter ===
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # === Price Position Relative to Daily Range ===
    # Normalized position: 0 = at low, 1 = at high, >1 = above high, <0 = below low
    range_size = high_1d_aligned - low_1d_aligned
    # Avoid division by zero
    range_size = np.where(range_size == 0, 1e-10, range_size)
    price_position = (close - low_1d_aligned) / range_size
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(high_1d_aligned[i]) or 
            np.isnan(low_1d_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or
            np.isnan(price_position[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price near daily low (<0.2) and above EMA50 (mean reversion in uptrend)
            if price_position[i] < 0.2 and close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price near daily high (>0.8) and below EMA50 (mean reversion in downtrend)
            elif price_position[i] > 0.8 and close[i] < ema50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price moves to upper half of daily range or trend breaks
            if price_position[i] > 0.6 or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price moves to lower half of daily range or trend breaks
            if price_position[i] < 0.4 or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals