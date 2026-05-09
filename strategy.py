#!/usr/bin/env python3
# Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume spike
# Long when price breaks above upper Donchian with EMA50 uptrend and volume > 1.5x average
# Short when price breaks below lower Donchian with EMA50 downtrend and volume > 1.5x average
# Exit when price crosses the 20-period EMA on 12h timeframe
# Uses Donchian channels for breakout signals, EMA for trend, volume for confirmation
# Designed to capture trend continuations in both bull and bear markets with controlled frequency
# Target: 50-150 total trades over 4 years (12-37/year) with size 0.25

name = "12h_Donchian_20_1dEMA50_VolumeSpike"
timeframe = "12h"
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
    
    # Calculate 20-period Donchian channels on 12h timeframe
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max()
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min()
    upper_donchian = high_roll.values
    lower_donchian = low_roll.values
    
    # Calculate 12h EMA20 for exit
    ema20_12h = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_confirm = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for calculations
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(upper_donchian[i]) or np.isnan(lower_donchian[i]) or 
            np.isnan(ema20_12h[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above upper Donchian, EMA50 uptrend, volume spike
            if (close[i] > upper_donchian[i] and 
                ema50_1d_aligned[i] > ema50_1d_aligned[i-1] and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower Donchian, EMA50 downtrend, volume spike
            elif (close[i] < lower_donchian[i] and 
                  ema50_1d_aligned[i] < ema50_1d_aligned[i-1] and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below 12h EMA20
            if close[i] < ema20_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above 12h EMA20
            if close[i] > ema20_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals