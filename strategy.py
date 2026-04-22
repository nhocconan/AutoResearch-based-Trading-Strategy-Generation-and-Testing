#!/usr/bin/env python3
"""
Hypothesis: 6-hour Donchian breakout with weekly pivot direction and volume confirmation.
Long when price breaks above Donchian(20) high and price > weekly pivot (bullish bias).
Short when price breaks below Donchian(20) low and price < weekly pivot (bearish bias).
Exit when price crosses back through Donchian middle or weekly pivot flips.
Weekly pivot provides structural bias, Donchian captures breakouts, volume confirms participation.
Designed for low trade frequency by requiring multiple confirmations and using higher timeframe bias.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    def rolling_max(arr, window):
        res = np.full_like(arr, np.nan, dtype=float)
        for i in range(window-1, len(arr)):
            res[i] = np.max(arr[i-window+1:i+1])
        return res
    
    def rolling_min(arr, window):
        res = np.full_like(arr, np.nan, dtype=float)
        for i in range(window-1, len(arr)):
            res[i] = np.min(arr[i-window+1:i+1])
        return res
    
    donchian_high = rolling_max(high, 20)
    donchian_low = rolling_min(low, 20)
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Load weekly data for pivot point - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Weekly pivot point: (H + L + C) / 3
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: break above Donchian high and price > weekly pivot with volume spike
            if (close[i] > donchian_high[i] and close[i] > weekly_pivot_aligned[i] and vol_spike):
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low and price < weekly pivot with volume spike
            elif (close[i] < donchian_low[i] and close[i] < weekly_pivot_aligned[i] and vol_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price crosses Donchian middle or weekly pivot flips
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses below Donchian mid or weekly pivot
                if close[i] < donchian_mid[i] or close[i] < weekly_pivot_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price crosses above Donchian mid or weekly pivot
                if close[i] > donchian_mid[i] or close[i] > weekly_pivot_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Donchian_WeeklyPivot_Volume"
timeframe = "6h"
leverage = 1.0