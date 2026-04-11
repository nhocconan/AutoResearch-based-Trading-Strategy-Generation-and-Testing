#!/usr/bin/env python3
# 6h_1w_52week_high_low_v1
# Strategy: 6h 52-week high/low breakout with volume confirmation
# Timeframe: 6h
# Leverage: 1.0
# Hypothesis: Weekly 52-week high/low levels act as strong support/resistance.
# Price breaking above 52-week high signals strength; breaking below 52-week low signals weakness.
# Volume confirmation ensures institutional participation. Works in both bull and bear markets:
# - In bull: buying breakouts above 52-week high
# - In bear: shorting breakdowns below 52-week low
# Using weekly timeframe for 52-week levels avoids noise from lower timeframes.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_52week_high_low_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 52:
        return np.zeros(n)
    
    # Calculate 52-week high and low
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Rolling 52-week high/low (52 weeks of data)
    week_high = pd.Series(high_1w).rolling(window=52, min_periods=52).max().values
    week_low = pd.Series(low_1w).rolling(window=52, min_periods=52).min().values
    
    # Align 52-week levels to 6h timeframe
    week_high_aligned = align_htf_to_ltf(prices, df_1w, week_high)
    week_low_aligned = align_htf_to_ltf(prices, df_1w, week_low)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_ratio = pd.Series(volume) / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(52, n):
        # Skip if any required data is invalid
        if (np.isnan(week_high_aligned[i]) or np.isnan(week_low_aligned[i]) or 
            np.isnan(vol_ratio.iloc[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.5x average
        vol_confirmed = vol_ratio.iloc[i] > 1.5
        
        # Entry conditions
        # Long: Price breaks above 52-week high + volume confirmation
        if vol_confirmed and close[i] > week_high_aligned[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: Price breaks below 52-week low + volume confirmation
        elif vol_confirmed and close[i] < week_low_aligned[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: price returns to midpoint of 52-week range or opposite breakout
        elif position == 1 and (close[i] < (week_high_aligned[i] + week_low_aligned[i]) / 2):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > (week_high_aligned[i] + week_low_aligned[i]) / 2):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals