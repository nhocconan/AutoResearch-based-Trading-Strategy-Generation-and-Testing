#!/usr/bin/env python3
# Hypothesis: 4h Donchian breakout with 1d EMA trend filter and volume confirmation
# Long when price breaks above Donchian(20) high, price > 1d EMA(50), and volume > 1.5x 20-period average
# Short when price breaks below Donchian(20) low, price < 1d EMA(50), and volume > 1.5x 20-period average
# Exit when price crosses Donchian midpoint or volume confirmation fails
# Uses Donchian for breakout signals, EMA for trend filter, volume for conviction
# Designed to capture strong trends while avoiding choppy markets with strict entry conditions
# Target: 100-180 total trades over 4 years (25-45/year) with size 0.25

name = "4h_Donchian_1dEMA_Volume_Breakout"
timeframe = "4h"
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
    
    # Calculate Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Calculate 1d EMA(50) for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_confirm = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for EMA calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above Donchian high, above 1d EMA, volume confirmation
            if (close[i] > donchian_high[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian low, below 1d EMA, volume confirmation
            elif (close[i] < donchian_low[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below Donchian midpoint or volume confirmation fails
            if (close[i] < donchian_mid[i]) or (not vol_confirm[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above Donchian midpoint or volume confirmation fails
            if (close[i] > donchian_mid[i]) or (not vol_confirm[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals