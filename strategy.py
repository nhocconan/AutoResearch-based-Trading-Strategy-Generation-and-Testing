#!/usr/bin/env python3
"""
12h_Donchian_Breakout_VolumeTrend_v1
Breakout above Donchian(20) high or below Donchian(20) low with volume confirmation and 1d EMA200 trend filter.
Exit when price returns to Donchian middle (midpoint of 20-period high-low).
Designed to capture breakouts in trending markets with volume confirmation.
Target: 50-150 total trades over 4 years (12-37/year).
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
    
    # === Donchian Channel (20) ===
    # Upper: highest high of last 20 periods
    # Lower: lowest low of last 20 periods
    # Middle: average of upper and lower
    def rolling_max(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            result[i] = np.max(arr[i-window+1:i+1])
        return result
    
    def rolling_min(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            result[i] = np.min(arr[i-window+1:i+1])
        return result
    
    donch_high = rolling_max(high, 20)
    donch_low = rolling_min(low, 20)
    donch_mid = (donch_high + donch_low) / 2.0
    
    # === Volume confirmation: volume > 1.5x 20-period average volume ===
    vol_ma = np.full_like(volume, np.nan)
    for i in range(19, len(volume)):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    volume_ratio = volume / (vol_ma + 1e-10)
    
    # === 1d EMA200 for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    ema_200_1d = pd.Series(df_1d['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    signals = np.zeros(n)
    
    # Warmup period: need 20 for Donchian, 19 for volume MA, 200 for EMA
    warmup = 200
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_high[i]) or 
            np.isnan(donch_low[i]) or 
            np.isnan(donch_mid[i]) or 
            np.isnan(volume_ratio[i]) or 
            np.isnan(ema_200_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price breaks above Donchian high, volume > 1.5x average, price above 1d EMA200
            if (close[i] > donch_high[i] and 
                volume_ratio[i] > 1.5 and 
                close[i] > ema_200_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below Donchian low, volume > 1.5x average, price below 1d EMA200
            elif (close[i] < donch_low[i] and 
                  volume_ratio[i] > 1.5 and 
                  close[i] < ema_200_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: return to Donchian middle
        elif position == 1:
            # Exit long: price crosses below Donchian middle
            if close[i] < donch_mid[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above Donchian middle
            if close[i] > donch_mid[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian_Breakout_VolumeTrend_v1"
timeframe = "12h"
leverage = 1.0