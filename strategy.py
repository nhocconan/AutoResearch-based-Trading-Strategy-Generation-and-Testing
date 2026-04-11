#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d/1w trend filter.
# Uses 1w SMA(13,8,5) for trend direction (Jaws/Teeth/Lips alignment).
# Enters long when 12h price crosses above Lips in bullish weekly trend.
# Enters short when 12h price crosses below Lips in bearish weekly trend.
# Volume filter (1.5x 20-period average) confirms momentum.
# Designed for 12-37 trades/year on 12h timeframe.

name = "12h_1w_alligator_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate Williams Alligator on weekly timeframe
    close_1w = df_1w['close'].values
    
    # Jaw (13-period, 8-bar shift)
    jaw = pd.Series(close_1w).rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth (8-period, 5-bar shift)
    teeth = pd.Series(close_1w).rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips (5-period, 3-bar shift)
    lips = pd.Series(close_1w).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Weekly trend: Lips > Teeth > Jaw = bullish, Lips < Teeth < Jaw = bearish
    weekly_trend_bull = (lips > teeth) & (teeth > jaw)
    weekly_trend_bear = (lips < teeth) & (teeth < jaw)
    
    # Align weekly Alligator lines and trend to 12h
    jaw_aligned = align_htf_to_ltf(prices, df_1w, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1w, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1w, lips)
    weekly_trend_bull_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_bull)
    weekly_trend_bear_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_bear)
    
    # Volume filter: 20-period average on 12h
    vol_avg_20 = np.full_like(volume, np.nan, dtype=float)
    for i in range(19, len(volume)):
        vol_avg_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(1, n):
        # Skip if any required data is invalid
        if (np.isnan(lips_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(jaw_aligned[i]) or
            np.isnan(weekly_trend_bull_aligned[i]) or np.isnan(weekly_trend_bear_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume filter: current volume > 1.5 * 20-period average
        vol_filter = volume[i] > 1.5 * vol_avg_20[i]
        
        # Determine weekly trend direction
        is_bullish_week = weekly_trend_bull_aligned[i]
        is_bearish_week = weekly_trend_bear_aligned[i]
        
        # Enter long when price crosses above Lips in bullish weekly trend
        long_entry = (close[i] > lips_aligned[i] and close[i-1] <= lips_aligned[i-1] and 
                     vol_filter and is_bullish_week)
        
        # Enter short when price crosses below Lips in bearish weekly trend
        short_entry = (close[i] < lips_aligned[i] and close[i-1] >= lips_aligned[i-1] and 
                      vol_filter and is_bearish_week)
        
        # Exit when price crosses Teeth (opposite signal)
        exit_long = (position == 1 and close[i] < teeth_aligned[i])
        exit_short = (position == -1 and close[i] > teeth_aligned[i])
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals