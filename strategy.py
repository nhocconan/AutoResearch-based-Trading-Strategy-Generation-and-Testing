#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 12h Donchian Breakout with Weekly Trend Filter
# Hypothesis: Trade breakouts of 20-period Donchian channels on 12h timeframe,
# filtered by weekly EMA(50) trend direction. Works in both bull and bear markets
# by only taking breakouts in the direction of the higher timeframe trend.
# Target: 50-150 total trades over 4 years (12-37/year) with low frequency to minimize fee drag.

name = "12h_donchian20_weekly_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    close_weekly = df_weekly['close'].values
    
    # Donchian Channel (20-period) on 12h
    donchian_period = 20
    upper_dc = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lower_dc = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Weekly EMA(50) for trend filter
    ema_50_weekly = pd.Series(close_weekly).ewm(span=50, adjust=False).mean().values
    ema_50_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_50_weekly)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(upper_dc[i]) or np.isnan(lower_dc[i]) or 
            np.isnan(ema_50_weekly_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below lower Donchian channel or trend changes to down
            if close[i] < lower_dc[i] or close[i] < ema_50_weekly_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price closes above upper Donchian channel or trend changes to up
            if close[i] > upper_dc[i] or close[i] > ema_50_weekly_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long breakout: price closes above upper Donchian with up trend
            if close[i] > upper_dc[i] and close[i] > ema_50_weekly_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short breakdown: price closes below lower Donchian with down trend
            elif close[i] < lower_dc[i] and close[i] < ema_50_weekly_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals