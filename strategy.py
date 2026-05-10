#!/usr/bin/env python3
# 1d_WeeklyDonchian_Breakout_Volume
# Hypothesis: Weekly Donchian channels capture long-term trend structure.
# Breakouts above weekly high or below weekly low with volume confirmation signal strong momentum.
# Volume filter ensures breakouts are supported by participation, reducing false signals.
# Designed for low trade frequency (10-25/year) to minimize fee drift on daily timeframe.
# Works in both bull and bear markets by capturing sustained moves in either direction.

name = "1d_WeeklyDonchian_Breakout_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data (HTF)
    df_weekly = get_htf_data(prices, '1w')
    
    # Weekly Donchian channels (20-week lookback)
    donch_period = 20
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    
    # Calculate weekly high and low over donch_period
    weekly_high = pd.Series(high_weekly).rolling(window=donch_period, min_periods=donch_period).max().values
    weekly_low = pd.Series(low_weekly).rolling(window=donch_period, min_periods=donch_period).min().values
    
    # Align to daily timeframe (waits for weekly bar to close)
    weekly_high_aligned = align_htf_to_ltf(prices, df_weekly, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_weekly, weekly_low)
    
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
    
    start_idx = max(60, donch_period)  # Need enough history
    
    for i in range(start_idx, n):
        if np.isnan(weekly_high_aligned[i]) or np.isnan(weekly_low_aligned[i]) or \
           np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_confirm = volume[i] > 1.5 * vol_ma[i] if vol_ma[i] > 0 else False
        
        if position == 0:
            # Long breakout: price closes above weekly high AND volume confirmation
            if close[i] > weekly_high_aligned[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short breakout: price closes below weekly low AND volume confirmation
            elif close[i] < weekly_low_aligned[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below weekly low (contrarian exit)
            if close[i] < weekly_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above weekly high (contrarian exit)
            if close[i] > weekly_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals