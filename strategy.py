#!/usr/bin/env python3
"""
Hypothesis: 6-hour Donchian Breakout with 12-hour Volume and 1-day Trend Filter.
Long when price breaks above Donchian(20) high, 12-hour volume > 1.5x average, and 1-day EMA34 rising.
Short when price breaks below Donchian(20) low, 12-hour volume > 1.5x average, and 1-day EMA34 falling.
Exit when price crosses Donchian midpoint or volume drops.
Uses volume confirmation to reduce false breakouts and daily trend filter to align with higher timeframe.
Designed for low trade frequency by requiring multiple confirmations.
Works in both bull and bear markets by following daily trend while using 6h Donchian for entries.
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
    
    # Load 1-day data for EMA34 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Load 12-hour data for volume average - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    volume_12h = df_12h['volume'].values
    vol_avg_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_avg_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_avg_12h)
    
    # Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_20 + low_20) / 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_avg_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above Donchian high, volume spike, and daily EMA rising
            if (high[i] > high_20[i-1] and  # Break above previous Donchian high
                volume[i] > 1.5 * vol_avg_12h_aligned[i] and
                ema34_1d_aligned[i] > ema34_1d_aligned[i-1]):
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian low, volume spike, and daily EMA falling
            elif (low[i] < low_20[i-1] and  # Break below previous Donchian low
                  volume[i] > 1.5 * vol_avg_12h_aligned[i] and
                  ema34_1d_aligned[i] < ema34_1d_aligned[i-1]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price falls below Donchian midpoint OR volume drops below average
                if (close[i] < donchian_mid[i] or 
                    volume[i] < vol_avg_12h_aligned[i]):
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price rises above Donchian midpoint OR volume drops below average
                if (close[i] > donchian_mid[i] or 
                    volume[i] < vol_avg_12h_aligned[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Donchian_Breakout_12hVol_1dEMA34_Trend"
timeframe = "6h"
leverage = 1.0