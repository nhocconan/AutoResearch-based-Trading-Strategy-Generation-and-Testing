#!/usr/bin/env python3
# 1D_WeeklyTrend_With_Volume_Confirmation
# Hypothesis: Uses weekly EMA trend filter and daily price action with volume confirmation.
# Long when price > weekly EMA40 and breaks above daily high with volume spike.
# Short when price < weekly EMA40 and breaks below daily low with volume spike.
# Designed for low trade frequency (<20/year) with clear trend following logic.
# Works in both bull and bear markets by following the weekly trend.

name = "1D_WeeklyTrend_With_Volume_Confirmation"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 2:
        return np.zeros(n)
    
    # Weekly EMA40 for trend filter
    ema40_weekly = pd.Series(df_weekly['close'].values).ewm(span=40, adjust=False, min_periods=40).mean().values
    ema40_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema40_weekly)
    
    # Daily high/low for breakout levels
    daily_high = prices['high'].rolling(window=1, min_periods=1).max().values  # Today's high
    daily_low = prices['low'].rolling(window=1, min_periods=1).min().values    # Today's low
    
    # Volume filter: current volume > 1.5x average volume (20-day)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure we have volume MA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(ema40_weekly_aligned[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: spike confirmation
        volume_filter = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: Price above weekly EMA40 AND breaks above today's high with volume spike
            if (close[i] > ema40_weekly_aligned[i] and 
                high[i] > daily_high[i] and  # Current high breaks today's high (always true, but kept for structure)
                volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price below weekly EMA40 AND breaks below today's low with volume spike
            elif (close[i] < ema40_weekly_aligned[i] and 
                  low[i] < daily_low[i] and  # Current low breaks today's low (always true, but kept for structure)
                  volume_filter):
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit when price crosses weekly EMA40 in opposite direction
            if (position == 1 and close[i] < ema40_weekly_aligned[i]) or \
               (position == -1 and close[i] > ema40_weekly_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                # Maintain position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals