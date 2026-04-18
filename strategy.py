#!/usr/bin/env python3
"""
6h_WeeklyPivot_DonchianBreakout_Trend
Hypothesis: 6-hour breakouts above weekly pivot-based Donchian channels with weekly trend filter.
In bull markets: captures breakout momentum above weekly resistance.
In bear markets: avoids false breakouts via weekly trend filter and captures breakdowns.
Weekly pivot provides structural support/resistance; Donchian(20) provides breakout signal.
Target: 50-150 total trades over 4 years (12-37/year).
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
    
    # Get weekly data for pivot and trend
    df_w = get_htf_data(prices, '1w')
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    close_w = df_w['close'].values
    
    # Weekly pivot point (standard calculation)
    pivot_w = (high_w + low_w + close_w) / 3
    
    # Weekly Donchian channel (20-period)
    def donchian_channel(high_arr, low_arr, period):
        upper = np.full_like(high_arr, np.nan)
        lower = np.full_like(low_arr, np.nan)
        for i in range(period-1, len(high_arr)):
            upper[i] = np.max(high_arr[i-period+1:i+1])
            lower[i] = np.min(low_arr[i-period+1:i+1])
        return upper, lower
    
    donchian_upper_w, donchian_lower_w = donchian_channel(high_w, low_w, 20)
    
    # Weekly EMA34 for trend filter
    def ema(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        multiplier = 2 / (period + 1)
        result[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            result[i] = arr[i] * multiplier + result[i-1] * (1 - multiplier)
        return result
    
    ema34_w = ema(close_w, 34)
    
    # Align weekly indicators to 6h timeframe
    pivot_w_aligned = align_htf_to_ltf(prices, df_w, pivot_w)
    donchian_upper_w_aligned = align_htf_to_ltf(prices, df_w, donchian_upper_w)
    donchian_lower_w_aligned = align_htf_to_ltf(prices, df_w, donchian_lower_w)
    ema34_w_aligned = align_htf_to_ltf(prices, df_w, ema34_w)
    
    # Volume confirmation: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_w_aligned[i]) or np.isnan(donchian_upper_w_aligned[i]) or 
            np.isnan(donchian_lower_w_aligned[i]) or np.isnan(ema34_w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above weekly Donchian upper with volume and weekly uptrend
            if (close[i] > donchian_upper_w_aligned[i] and vol_confirm[i] and 
                close[i] > ema34_w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below weekly Donchian lower with volume and weekly downtrend
            elif (close[i] < donchian_lower_w_aligned[i] and vol_confirm[i] and 
                  close[i] < ema34_w_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below weekly pivot or weekly trend turns down
            if (close[i] < pivot_w_aligned[i] or close[i] < ema34_w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above weekly pivot or weekly trend turns up
            if (close[i] > pivot_w_aligned[i] or close[i] > ema34_w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_DonchianBreakout_Trend"
timeframe = "6h"
leverage = 1.0