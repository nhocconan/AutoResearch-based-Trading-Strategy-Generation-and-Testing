#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian breakout with 12h EMA trend filter and volume spike confirmation.
Long when price breaks above Donchian(20) high + 12h EMA50 up + volume > 2x average.
Short when price breaks below Donchian(20) low + 12h EMA50 down + volume > 2x average.
Exit when price returns to Donchian midpoint or trend changes.
Designed for moderate trade frequency (~20-40/year) with strong trend capture in both bull and bear markets.
"""

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
    
    # Load 12h data for trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high + lowest_low) / 2
    
    # Calculate average volume for confirmation
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(avg_volume[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Breakout above Donchian high + 12h EMA up + volume spike
            if (close[i] > highest_high[i] and 
                ema_50_12h_aligned[i] > close_12h[-1] if len(close_12h) > 0 else False and  # Previous 12h close < current EMA (uptrend)
                volume[i] > 2.0 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # Short: Breakout below Donchian low + 12h EMA down + volume spike
            elif (close[i] < lowest_low[i] and 
                  ema_50_12h_aligned[i] < close_12h[-1] if len(close_12h) > 0 else False and  # Previous 12h close > current EMA (downtrend)
                  volume[i] > 2.0 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price returns to midpoint or 12h EMA turns down
                if close[i] <= donchian_mid[i] or ema_50_12h_aligned[i] < close_12h[-1] if len(close_12h) > 0 else False:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price returns to midpoint or 12h EMA turns up
                if close[i] >= donchian_mid[i] or ema_50_12h_aligned[i] > close_12h[-1] if len(close_12h) > 0 else False:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian_Breakout_12hEMA_VolumeSpike"
timeframe = "4h"
leverage = 1.0