#!/usr/bin/env python3
# Hypothesis: 1-day Donchian(20) breakout with 1-week EMA50 trend filter and volume confirmation
# Long when price breaks above 1-week EMA50 and closes above Donchian high with volume > 1.5x average
# Short when price breaks below 1-week EMA50 and closes below Donchian low with volume > 1.5x average
# Exit when price crosses back over 1-week EMA50
# Uses weekly EMA for trend direction, daily Donchian for breakout signals, volume for confirmation
# Designed to capture trend continuation in both bull and bear markets with controlled frequency
# Target: 30-100 total trades over 4 years (7-25/year) with size 0.25

name = "1d_Donchian_20_1wEMA50_VolumeConfirmation"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need enough data for 20-period Donchian
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1-week EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate daily Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Need enough data for Donchian calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price above EMA50 and breaks above Donchian high with volume confirmation
            if (close[i] > ema50_1w_aligned[i] and 
                close[i] > donchian_high[i] and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price below EMA50 and breaks below Donchian low with volume confirmation
            elif (close[i] < ema50_1w_aligned[i] and 
                  close[i] < donchian_low[i] and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses back below EMA50
            if close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses back above EMA50
            if close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals