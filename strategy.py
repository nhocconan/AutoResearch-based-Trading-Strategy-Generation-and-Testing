#!/usr/bin/env python3
"""
1d_WilliamsAlligator_Trend_1wEMA34_HTFTrend
Hypothesis: Daily Williams Alligator (jaw/teeth/lips) with 1-week EMA34 trend filter.
Enters long when Alligator is bullish (lips>teeth>jaw) and price > 1w EMA34.
Enters short when Alligator is bearish (lips<teeth<jaw) and price < 1w EMA34.
Uses discrete position sizing (0.0, ±0.25) to minimize fee churn. Designed for 30-100 total trades over 4 years.
Works in both bull and bear markets by following the 1w trend direction only.
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
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate Williams Alligator on daily timeframe
    # Jaw: 13-period SMMA, shifted by 8
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().values
    jaw = np.roll(jaw, 8)
    jaw[:8] = np.nan
    
    # Teeth: 8-period SMMA, shifted by 5
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth, 5)
    teeth[:5] = np.nan
    
    # Lips: 5-period SMMA, shifted by 3
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips, 3)
    lips[:3] = np.nan
    
    # Calculate 1-week EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need Alligator warmup: 13+8=21)
    start_idx = 21
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or 
            np.isnan(ema_34_1w_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Bullish Alligator: lips > teeth > jaw
        bullish = lips[i] > teeth[i] and teeth[i] > jaw[i]
        # Bearish Alligator: lips < teeth < jaw
        bearish = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        # Long logic: bullish Alligator + price > 1w EMA34
        if bullish and close[i] > ema_34_1w_aligned[i]:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: bearish Alligator + price < 1w EMA34
        elif bearish and close[i] < ema_34_1w_aligned[i]:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit: Alligator changes direction
        elif position == 1 and not bullish:
            signals[i] = 0.0
            position = 0
        elif position == -1 and not bearish:
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

name = "1d_WilliamsAlligator_Trend_1wEMA34_HTFTrend"
timeframe = "1d"
leverage = 1.0