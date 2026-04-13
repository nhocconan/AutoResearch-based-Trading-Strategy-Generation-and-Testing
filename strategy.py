#!/usr/bin/env python3
"""
4h_1d_Midpoint_Bounce_With_Trend
Hypothesis: Price tends to bounce off the daily midpoint (H+L)/2 in trending markets.
We take long when price > daily midpoint and 4h close > 4h EMA20, short when price < daily midpoint and 4h close < 4h EMA20.
Works in bull markets (buying dips) and bear markets (selling rallies) by following the 4h trend.
Target: 20-30 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get daily data for midpoint
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily midpoint: (H+L)/2
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    midpoint_1d = (high_1d + low_1d) / 2.0
    
    # Align midpoint to 4h timeframe
    midpoint_aligned = align_htf_to_ltf(prices, df_1d, midpoint_1d)
    
    # 4h EMA20 for trend filter
    close_series = pd.Series(close)
    ema_20 = close_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(100, n):
        # Skip if any required data is not ready
        if np.isnan(midpoint_aligned[i]) or np.isnan(ema_20[i]):
            signals[i] = 0.0
            continue
        
        # Long condition: price above daily midpoint AND 4h close above EMA20
        long_condition = close[i] > midpoint_aligned[i] and close[i] > ema_20[i]
        
        # Short condition: price below daily midpoint AND 4h close below EMA20
        short_condition = close[i] < midpoint_aligned[i] and close[i] < ema_20[i]
        
        if long_condition and position != 1:
            position = 1
            signals[i] = position_size
        elif short_condition and position != -1:
            position = -1
            signals[i] = -position_size
        elif not long_condition and not short_condition and position != 0:
            # Exit when conditions not met
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "4h_1d_Midpoint_Bounce_With_Trend"
timeframe = "4h"
leverage = 1.0