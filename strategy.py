#!/usr/bin/env python3
"""
Hypothesis: 4-hour Donchian channel breakout with 12-hour EMA trend filter and volume confirmation.
Long when price breaks above Donchian(20) high, 12-hour EMA50 is rising, and volume > 1.5x average.
Short when price breaks below Donchian(20) low, 12-hour EMA50 is falling, and volume > 1.5x average.
Exit when price returns to Donchian midpoint or EMA trend reverses.
This strategy targets low trade frequency by requiring multiple confirmations and works in both bull and bear markets by following the 12-hour trend.
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
    
    # Load 12-hour data for EMA50 trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Donchian channel (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_20 + low_20) / 2.0
    
    # Volume average (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(vol_avg[i]) or
            np.isnan(ema50_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian high, EMA50 rising, volume spike
            if (close[i] > high_20[i] and 
                ema50_12h_aligned[i] > ema50_12h_aligned[i-1] and
                volume[i] > 1.5 * vol_avg[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low, EMA50 falling, volume spike
            elif (close[i] < low_20[i] and 
                  ema50_12h_aligned[i] < ema50_12h_aligned[i-1] and
                  volume[i] > 1.5 * vol_avg[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price returns to midpoint OR EMA trend turns down
                if (close[i] <= donchian_mid[i] or 
                    ema50_12h_aligned[i] < ema50_12h_aligned[i-1]):
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price returns to midpoint OR EMA trend turns up
                if (close[i] >= donchian_mid[i] or 
                    ema50_12h_aligned[i] > ema50_12h_aligned[i-1]):
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