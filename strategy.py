#!/usr/bin/env python3
"""
6h Williams Alligator + Weekly EMA Trend Filter
Hypothesis: Williams Alligator (Jaw/Teeth/Lips) identifies trend absence/presence.
When aligned with weekly EMA50 trend, it captures strong moves in both bull and bear markets.
Uses 6h timeframe with 1w HTF for trend filter. Targets 50-150 trades over 4 years.
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
    
    # Get 1w data for weekly EMA50 trend (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 50-period EMA on weekly
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Williams Alligator on 6h: SMAs shifted forward
    # Jaw: 13-period SMA, shifted 8 bars forward
    # Teeth: 8-period SMA, shifted 5 bars forward  
    # Lips: 5-period SMA, shifted 3 bars forward
    jaw = pd.Series(high).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(low).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Alligator shifts
    start_idx = 13  # max shift is 8, but need 13 for jaw calculation
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Alligator conditions:
        # Mouth open (trending): Lips > Teeth > Jaw (bullish) OR Lips < Teeth < Jaw (bearish)
        # Mouth closed (ranging): otherwise
        bullish_aligned = lips[i] > teeth[i] and teeth[i] > jaw[i]
        bearish_aligned = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        weekly_trend_up = close[i] > ema_50_1w_aligned[i]
        weekly_trend_down = close[i] < ema_50_1w_aligned[i]
        
        if position == 0:
            # Long: bullish Alligator alignment + weekly uptrend
            if bullish_aligned and weekly_trend_up:
                signals[i] = 0.25
                position = 1
            # Short: bearish Alligator alignment + weekly downtrend
            elif bearish_aligned and weekly_trend_down:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: exit when Alligator closes or weekly trend turns down
            if not (bullish_aligned and weekly_trend_up):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short: exit when Alligator closes or weekly trend turns up
            if not (bearish_aligned and weekly_trend_down):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Williams_Alligator_WeeklyEMA_Trend_Filter"
timeframe = "6h"
leverage = 1.0