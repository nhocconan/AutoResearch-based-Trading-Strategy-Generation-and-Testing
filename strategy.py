#!/usr/bin/env python3
"""
1d_Williams_Alligator_1wTrend_Filter_v1
Hypothesis: Williams Alligator (jaw=EMA13, teeth=EMA8, lips=EMA5) on daily timeframe with 1-week EMA50 trend filter.
Only long when lips > teeth > jaw (bullish alignment) and price > 1w EMA50.
Only short when lips < teeth < jaw (bearish alignment) and price < 1w EMA50.
Uses discrete position sizing (0.0, ±0.25) to minimize fee churn. Target: 20-80 total trades over 4 years (5-20/year).
Designed to work in both bull and bear markets by combining Alligator's trend identification with higher timeframe trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate Williams Alligator components (5,8,13 period SMAs of median price)
    median_price = (high + low) / 2
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values  # EMA13 equivalent via SMA for simplicity
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values    # EMA8
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values     # EMA5
    
    # Load 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(13, 8, 5, 50) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_1w_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Discrete position sizing
        base_size = 0.25
        
        # Bullish Alligator alignment: lips > teeth > jaw
        bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
        # Bearish Alligator alignment: lips < teeth < jaw
        bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        # Long logic: Bullish alignment + price > 1w EMA50 (uptrend)
        if bullish_alignment and close[i] > ema_50_1w_aligned[i]:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: Bearish alignment + price < 1w EMA50 (downtrend)
        elif bearish_alignment and close[i] < ema_50_1w_aligned[i]:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit conditions: loss of Alligator alignment or trend filter failure
        elif position == 1 and (not bullish_alignment or close[i] <= ema_50_1w_aligned[i]):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (not bearish_alignment or close[i] >= ema_50_1w_aligned[i]):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "1d_Williams_Alligator_1wTrend_Filter_v1"
timeframe = "1d"
leverage = 1.0