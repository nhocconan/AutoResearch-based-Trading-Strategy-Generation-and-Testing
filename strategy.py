#!/usr/bin/env python3
"""
6h_1d_alligator_ema_cross_v1
Hypothesis: Combine Williams Alligator (13,8,5 SMAs) with EMA(50) trend filter on 6b timeframe.
Alligator jaws (13SMA) acts as dynamic support/resistance. Teeth (8SMA) and lips (5SMA) convergence/divergence signals trend strength.
In bull markets: buy when lips cross above teeth with price above EMA50.
In bear markets: sell when lips cross below teeth with price below EMA50.
Uses 1d timeframe for Alligator calculation to avoid noise, aligned to 6b.
Target: 20-60 trades/year to minimize fee drag.
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
    
    # Get daily data for Alligator calculation (less noise)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Williams Alligator: 13, 8, 5 period SMAs of median price
    # Jaw (13), Teeth (8), Lips (5)
    median_price_1d = (high_1d + low_1d) / 2
    
    jaw = pd.Series(median_price_1d).rolling(window=13, min_periods=13).mean().values  # 13-period
    teeth = pd.Series(median_price_1d).rolling(window=8, min_periods=8).mean().values   # 8-period
    lips = pd.Series(median_price_1d).rolling(window=5, min_periods=5).mean().values    # 5-period
    
    # EMA(50) for trend filter on 1d
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Alligator conditions: lips > teeth > jaw = bullish alignment
    # lips < teeth < jaw = bearish alignment
    bullish_alignment = (lips > teeth) & (teeth > jaw)
    bearish_alignment = (lips < teeth) & (teeth < jaw)
    
    # Price relative to Alligator jaws (13SMA) - acts as support/resistance
    price_above_jaw = close_1d > jaw
    price_below_jaw = close_1d < jaw
    
    # Align all signals to 6h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    bullish_aligned = align_htf_to_ltf(prices, df_1d, bullish_alignment.astype(float))
    bearish_aligned = align_htf_to_ltf(prices, df_1d, bearish_alignment.astype(float))
    price_above_jaw_aligned = align_htf_to_ltf(prices, df_1d, price_above_jaw.astype(float))
    price_below_jaw_aligned = align_htf_to_ltf(prices, df_1d, price_below_jaw.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or 
            np.isnan(bullish_aligned[i]) or 
            np.isnan(bearish_aligned[i]) or 
            np.isnan(price_above_jaw_aligned[i]) or 
            np.isnan(price_below_jaw_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions
        long_entry = (bullish_aligned[i] > 0.5 and 
                     price_above_jaw_aligned[i] > 0.5 and 
                     close[i] > ema_50_aligned[i])
        
        short_entry = (bearish_aligned[i] > 0.5 and 
                      price_below_jaw_aligned[i] > 0.5 and 
                      close[i] < ema_50_aligned[i])
        
        # Exit when Alligator lines cross (trend change)
        exit_long = (lips_aligned[i] < teeth_aligned[i]) and (position == 1)
        exit_short = (lips_aligned[i] > teeth_aligned[i]) and (position == -1)
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1d_alligator_ema_cross_v1"
timeframe = "6h"
leverage = 1.0