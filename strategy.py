#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h_1d_1w_alligator_elder_ray_v1
# Uses weekly trend via Alligator (SMAs of median price) and daily Elder Ray (bull/bear power) for entry timing.
# Long when weekly trend is bullish (price > Alligator's Jaw) AND daily bull power > 0.
# Short when weekly trend is bearish (price < Alligator's Jaw) AND daily bear power < 0.
# Exits when Elder Ray signal reverses or price crosses Alligator's Teeth.
# Designed for low trade frequency (target: 12-37/year) with trend-following in weekly timeframe and precise entry on daily.
# Works in trending markets via Alligator filter and avoids whipsaws via Elder Ray confirmation.

name = "6h_1d_1w_alligator_elder_ray_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for Alligator
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 13:
        return np.zeros(n)
    
    # Get daily data for Elder Ray
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Calculate weekly median price (typical price)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    median_price_1w = (high_1w + low_1w + close_1w) / 3.0
    
    # Alligator: SMAs of median price (13, 8, 5) with future shift
    jaw = pd.Series(median_price_1w).rolling(window=13, min_periods=13).mean().values  # 13-period
    teeth = pd.Series(median_price_1w).rolling(window=8, min_periods=8).mean().values   # 8-period
    lips = pd.Series(median_price_1w).rolling(window=5, min_periods=5).mean().values    # 5-period
    
    # Align weekly Alligator to 6h timeframe (with proper delay for completed weekly bar)
    jaw_aligned = align_htf_to_ltf(prices, df_1w, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1w, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1w, lips)
    
    # Calculate daily Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high_1d - ema13_1d
    bear_power = low_1d - ema13_1d
    
    # Align daily Elder Ray to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # start after warmup
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Weekly trend filter: price vs Jaw (Alligator's Jaw)
        price_vs_jaw = close[i] - jaw_aligned[i]
        
        # Long setup: weekly bullish trend AND daily bull power positive
        if price_vs_jaw > 0 and bull_power_aligned[i] > 0 and position != 1:
            position = 1
            signals[i] = 0.25
        # Short setup: weekly bearish trend AND daily bear power negative
        elif price_vs_jaw < 0 and bear_power_aligned[i] < 0 and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: Elder Ray signal reverses OR price crosses Teeth
        elif position == 1 and (bull_power_aligned[i] <= 0 or close[i] <= teeth_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (bear_power_aligned[i] >= 0 or close[i] >= teeth_aligned[i]):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals