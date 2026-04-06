#!/usr/bin/env python3
"""
6h Alligator with Elder Ray Momentum and 12h Trend Filter
Hypothesis: The Alligator indicator identifies trend presence, while Elder Ray measures bull/bear power.
Combined with 12h trend filter, this captures strong trends while avoiding chop. Works in bull (long above teeth) and bear (short below teeth).
Target: 80-160 total trades over 4 years (20-40/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_alligator_elder_ray_12h_trend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 12h data for trend filter (once before loop)
    df_12h = get_htf_data(prices, '12h')
    
    # 12h EMA(50) for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Alligator: Jaw(13,8), Teeth(8,5), Lips(5,3)
    # Jaw: SMMA(13, 8)
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean()
    jaw = jaw.shift(8)
    jaw_values = jaw.values
    
    # Teeth: SMMA(8, 5)
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean()
    teeth = teeth.shift(5)
    teeth_values = teeth.values
    
    # Lips: SMMA(5, 3)
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean()
    lips = lips.shift(3)
    lips_values = lips.values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 13  # For EMA13 and Alligator
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(jaw_values[i]) or np.isnan(teeth_values[i]) or np.isnan(lips_values[i]) or
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: price below teeth OR bear power negative
            if (close[i] < teeth_values[i] or bear_power[i] < 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price above teeth OR bull power positive
            if (close[i] > teeth_values[i] or bull_power[i] > 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Alligator aligned + Elder Ray + 12h trend
            # Alligator aligned: Lips > Teeth > Jaw (uptrend) or Lips < Teeth < Jaw (downtrend)
            alligator_long = lips_values[i] > teeth_values[i] and teeth_values[i] > jaw_values[i]
            alligator_short = lips_values[i] < teeth_values[i] and teeth_values[i] < jaw_values[i]
            
            # 12h trend filter: price above/below EMA50
            price_above_12h = close[i] > ema_50_12h_aligned[i]
            price_below_12h = close[i] < ema_50_12h_aligned[i]
            
            long_setup = alligator_long and bull_power[i] > 0 and price_above_12h
            short_setup = alligator_short and bear_power[i] < 0 and price_below_12h
            
            if long_setup:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_setup:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals