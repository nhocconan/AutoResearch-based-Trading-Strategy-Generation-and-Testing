#!/usr/bin/env python3
"""
6h_RangeBreakout_TrailingVolume
Hypothesis: Range breakout strategy for 6h timeframe using 120-bar (30-day) highest/lowest levels as dynamic support/resistance. 
Entry on breakout with volume confirmation (>1.5x 20-bar average). Exit on close crossing back into the range or trailing stop. 
Uses 1-week EMA200 for trend filter to avoid counter-trend trades. Designed for low trade frequency (15-30/year) to minimize fee drag.
Works in both bull (breaks highs) and bear (breaks lows) markets by trading breakouts in direction of higher timeframe trend.
"""

name = "6h_RangeBreakout_TrailingVolume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 120-bar (30-day) highest high and lowest low for range
    highest_high = pd.Series(high).rolling(window=120, min_periods=120).max().values
    lowest_low = pd.Series(low).rolling(window=120, min_periods=120).min().values
    
    # Volume average for spike detection (20-bar)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get weekly EMA200 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    ema_200_1w = pd.Series(df_1w['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(120, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(ema_200_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike condition: current volume > 1.5x 20-bar average
        vol_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # LONG: Break above 120-period high with volume + above weekly EMA200
            if close[i] > highest_high[i] and vol_spike and close[i] > ema_200_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Break below 120-period low with volume + below weekly EMA200
            elif close[i] < lowest_low[i] and vol_spike and close[i] < ema_200_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close back below 120-period high OR below weekly EMA200
            if close[i] < highest_high[i] or close[i] < ema_200_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close back above 120-period low OR above weekly EMA200
            if close[i] > lowest_low[i] or close[i] > ema_200_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals