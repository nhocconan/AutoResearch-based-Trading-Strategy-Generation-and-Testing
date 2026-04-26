#!/usr/bin/env python3
"""
1d_WilliamsAlligator_Trend_1wEMA34_HTFTrend
Hypothesis: Daily Williams Alligator (jaw/teeth/lips) with weekly EMA34 trend filter.
Enters long when Alligator is bullish (lips>teeth>jaw) and weekly trend is up (close>EMA34).
Enters short when Alligator is bearish (lips<teeth<jaw) and weekly trend is down (close<EMA34).
Uses discrete position sizing (0.0, ±0.25) to minimize fee churn. Designed for 30-100 total trades over 4 years.
Williams Alligator catches trends early; weekly filter avoids counter-trend whipsaws in bear markets.
Works in both bull and bear markets by only trading in direction of weekly trend.
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
    
    # Williams Alligator on 1d timeframe (uses SMAs with specific periods)
    # Jaw: 13-period SMMA, shifted 8 bars forward
    # Teeth: 8-period SMMA, shifted 5 bars forward  
    # Lips: 5-period SMMA, shifted 3 bars forward
    # SMMA = smoothed moving average (similar to EMA but different alpha)
    # We'll approximate SMMA with EMA for simplicity (common practice)
    
    # Calculate SMMA-like values using EMA (acceptable approximation)
    jaw = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean()
    teeth = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean()
    lips = pd.Series(close).ewm(span=5, adjust=False, min_periods=5).mean()
    
    # Apply the shifts (Alligator specific)
    jaw_shifted = jaw.shift(8)
    teeth_shifted = teeth.shift(5)
    lips_shifted = lips.shift(3)
    
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need lips shift 3 + weekly EMA warmup)
    start_idx = max(13, 8, 5) + 8  # jaw shift 8
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw_shifted.iloc[i]) or np.isnan(teeth_shifted.iloc[i]) or 
            np.isnan(lips_shifted.iloc[i]) or np.isnan(ema_34_1w_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Alligator conditions
        jaw_val = jaw_shifted.iloc[i]
        teeth_val = teeth_shifted.iloc[i]
        lips_val = lips_shifted.iloc[i]
        
        # Bullish Alligator: lips > teeth > jaw
        bullish_alligator = lips_val > teeth_val and teeth_val > jaw_val
        # Bearish Alligator: lips < teeth < jaw  
        bearish_alligator = lips_val < teeth_val and teeth_val < jaw_val
        
        # Weekly trend filter
        weekly_uptrend = close[i] > ema_34_1w_aligned[i]
        weekly_downtrend = close[i] < ema_34_1w_aligned[i]
        
        # Long logic: bullish Alligator + weekly uptrend
        if bullish_alligator and weekly_uptrend:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: bearish Alligator + weekly downtrend
        elif bearish_alligator and weekly_downtrend:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit: Alligator reverses or weekly trend changes
        elif position == 1 and (not bullish_alligator or not weekly_uptrend):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (not bearish_alligator or not weekly_downtrend):
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