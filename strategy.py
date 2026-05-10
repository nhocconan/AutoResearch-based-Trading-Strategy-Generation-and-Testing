#!/usr/bin/env python3
# 1d_WeeklyDonchian_Breakout_Volume
# Hypothesis: Weekly Donchian breakouts capture major trend changes. Volume confirms institutional participation.
# Long when price breaks above weekly Donchian high with volume > 1.5x 20-day average.
# Short when price breaks below weekly Donchian low with volume > 1.5x 20-day average.
# Exit on opposite Donchian breakout or volume failure.
# Weekly timeframe reduces noise, daily provides timely entry. Target: 15-25 trades/year.

name = "1d_WeeklyDonchian_Breakout_Volume"
timeframe = "1d"
leverage = 1.0

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
    
    # Get weekly data ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) == 0:
        return np.zeros(n)
    
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    
    # Calculate Donchian channels (20-week lookback)
    def rolling_max(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            res[i] = np.max(arr[i-window+1:i+1])
        return res
    
    def rolling_min(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            res[i] = np.min(arr[i-window+1:i+1])
        return res
    
    weekly_donchian_high = rolling_max(weekly_high, 20)
    weekly_donchian_low = rolling_min(weekly_low, 20)
    
    # Align to daily timeframe (only use after weekly bar closes)
    donchian_high_aligned = align_htf_to_ltf(prices, df_weekly, weekly_donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_weekly, weekly_donchian_low)
    
    # Volume confirmation (20-day average)
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p-1, len(arr)):
                res[i] = np.mean(arr[i-p+1:i+1])
        return res
    
    vol_ma = mean_arr(volume, 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Need 20 weeks of history
    
    for i in range(start_idx, n):
        if np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or \
           np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_confirm = volume[i] > 1.5 * vol_ma[i] if vol_ma[i] > 0 else False
        
        if position == 0:
            # Long breakout: price above weekly Donchian high with volume
            if close[i] > donchian_high_aligned[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short breakout: price below weekly Donchian low with volume
            elif close[i] < donchian_low_aligned[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below weekly Donchian low OR volume fails
            if close[i] < donchian_low_aligned[i] or not vol_confirm:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above weekly Donchian high OR volume fails
            if close[i] > donchian_high_aligned[i] or not vol_confirm:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals