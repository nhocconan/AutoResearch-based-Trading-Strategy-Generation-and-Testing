#!/usr/bin/env python3
"""
1d_donchian_20_breakout_1w_trend_volume_v1
Hypothesis: Daily Donchian(20) breakouts with weekly trend filter (EMA20) and volume confirmation.
In long: price breaks above 20-day high with volume above average and price above weekly EMA20.
In short: price breaks below 20-day low with volume above average and price below weekly EMA20.
Uses Donchian channels for breakout structure, EMA for trend filter, and volume for confirmation.
Designed for 15-25 trades/year on 1d timeframe with clear breakout logic that works in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian_20_breakout_1w_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian(20) channels on daily
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    ema20_1w = pd.Series(df_1w['close'].values).ewm(span=20, adjust=False).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Volume confirmation: 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if data not available
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(ema20_1w_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirmed = volume[i] > vol_ma[i]
        
        # Donchian breakout conditions
        breakout_up = close[i] > high_roll[i-1]  # Break above previous 20-day high
        breakout_down = close[i] < low_roll[i-1]  # Break below previous 20-day low
        
        # Weekly trend filter
        above_weekly_ema20 = close[i] > ema20_1w_aligned[i]
        below_weekly_ema20 = close[i] < ema20_1w_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below 20-day low or trend turns bearish
            if close[i] < low_roll[i] or below_weekly_ema20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above 20-day high or trend turns bullish
            if close[i] > high_roll[i] or above_weekly_ema20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: breakout up with volume confirmation and bullish weekly trend
            if breakout_up and vol_confirmed and above_weekly_ema20:
                position = 1
                signals[i] = 0.25
            # Short: breakout down with volume confirmation and bearish weekly trend
            elif breakout_down and vol_confirmed and below_weekly_ema20:
                position = -1
                signals[i] = -0.25
    
    return signals