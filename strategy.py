#!/usr/bin/env python3
"""
Hypothesis: Daily Donchian Channel Breakout with Weekly Trend Filter and Volume Confirmation.
Long when price breaks above the 20-day high during weekly uptrend with volume spike.
Short when price breaks below the 20-day low during weekly downtrend with volume spike.
Exit when price returns to the 20-day midpoint or trend reverses.
Designed for low trade frequency (~10-30 trades/year) by requiring weekly trend alignment and volume confirmation.
Works in both bull and bear markets by following the weekly trend.
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
    
    # 20-day Donchian channel
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    mid_20 = (high_20 + low_20) / 2
    
    # Load weekly data for trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # 20-period EMA on weekly close for trend
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Volume confirmation: current volume > 1.5x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(40, n):
        # Skip if data not ready
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema20_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above 20-day high + weekly uptrend + volume spike
            if close[i] > high_20[i] and ema20_1w_aligned[i] > ema20_1w_aligned[i-1] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-day low + weekly downtrend + volume spike
            elif close[i] < low_20[i] and ema20_1w_aligned[i] < ema20_1w_aligned[i-1] and vol_spike:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to 20-day midpoint or trend reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: price below midpoint or weekly trend turns down
                if close[i] < mid_20[i] or ema20_1w_aligned[i] < ema20_1w_aligned[i-1]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price above midpoint or weekly trend turns up
                if close[i] > mid_20[i] or ema20_1w_aligned[i] > ema20_1w_aligned[i-1]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "Daily_Donchian_Breakout_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0