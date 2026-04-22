#!/usr/bin/env python3
"""
Hypothesis: 4-hour Donchian breakout with 12-hour EMA trend filter and volume confirmation.
Long when price breaks above 20-period Donchian high with price > 12h EMA50 and volume > 1.5x 20-period average volume.
Short when price breaks below 20-period Donchian low with price < 12h EMA50 and volume > 1.5x 20-period average volume.
Exit when price returns to the Donchian midpoint (mean of 20-period high and low).
Designed for low trade frequency (~20-40/year) to avoid fee drag while capturing trends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_20 + low_20) / 2.0
    
    # Load 12-hour data for EMA50 trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50
    close_12h = df_12h['close'].values
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # Volume filter: 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(donchian_mid[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian high with trend and volume confirmation
            if close[i] > high_20[i] and close[i] > ema_50_aligned[i] and volume[i] > 1.5 * vol_ma[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low with trend and volume confirmation
            elif close[i] < low_20[i] and close[i] < ema_50_aligned[i] and volume[i] > 1.5 * vol_ma[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price returns to Donchian midpoint
                if close[i] <= donchian_mid[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price returns to Donchian midpoint
                if close[i] >= donchian_mid[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian_Breakout_12hEMA50_Volume"
timeframe = "4h"
leverage = 1.0