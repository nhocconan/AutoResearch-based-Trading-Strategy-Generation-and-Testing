#!/usr/bin/env python3
"""
Hypothesis: 6-hour Price Channel Breakout with 12-hour Trend Filter.
Long when price breaks above 6-hour Donchian(20) high, 12-hour EMA50 is rising, and volume > 1.5x average.
Short when price breaks below 6-hour Donchian(20) low, 12-hour EMA50 is falling, and volume > 1.5x average.
Exit when price crosses back through the Donchian midpoint or volume dries up.
Uses price channels for structure, EMA50 for higher timeframe trend, and volume for confirmation.
Designed for low trade frequency by requiring multiple confirmations and avoiding whipsaws.
Works in bull markets by following uptrend breaks and in bear markets by following downtrend breaks.
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
    
    # 6-hour Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_20 + low_20) / 2.0
    
    # Volume average (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(donchian_mid[i]) or
            np.isnan(vol_avg[i]) or np.isnan(ema50_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above Donchian high, 12h EMA50 rising, volume spike
            if (close[i] > high_20[i] and 
                ema50_12h_aligned[i] > ema50_12h_aligned[i-1] and
                volume[i] > 1.5 * vol_avg[i]):
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian low, 12h EMA50 falling, volume spike
            elif (close[i] < low_20[i] and 
                  ema50_12h_aligned[i] < ema50_12h_aligned[i-1] and
                  volume[i] > 1.5 * vol_avg[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price falls back below Donchian midpoint OR volume dries up
                if (close[i] < donchian_mid[i] or 
                    volume[i] < 0.5 * vol_avg[i]):
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price rises back above Donchian midpoint OR volume dries up
                if (close[i] > donchian_mid[i] or 
                    volume[i] < 0.5 * vol_avg[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Donchian_Breakout_12hEMA50_Volume"
timeframe = "6h"
leverage = 1.0