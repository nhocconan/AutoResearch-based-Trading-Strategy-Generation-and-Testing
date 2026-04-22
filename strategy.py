#!/usr/bin/env python3
"""
Hypothesis: 1-day Donchian breakout with 1-week trend filter and volume confirmation.
Long when price breaks above the 20-day high during 1-week uptrend with volume spike.
Short when price breaks below the 20-day low during 1-week downtrend with volume spike.
Exit when price returns to the 20-day midpoint or trend reverses.
Designed for low trade frequency by requiring trend alignment and volume confirmation.
Works in both bull and bear markets by following the 1-week trend.
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
    
    # 20-day Donchian channels
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    mid_20 = (high_20 + low_20) / 2
    
    # Load 1-week data for trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # 20-period EMA on 1w close for trend
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Volume confirmation: current volume > 1.5x 20-period average
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
            # Long: price breaks above 20-day high + 1w uptrend + volume spike
            if close[i] > high_20[i] and ema20_1w_aligned[i] > ema20_1w_aligned[i-1] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-day low + 1w downtrend + volume spike
            elif close[i] < low_20[i] and ema20_1w_aligned[i] < ema20_1w_aligned[i-1] and vol_spike:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to 20-day midpoint or trend reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: price below midpoint or 1w trend turns down
                if close[i] < mid_20[i] or ema20_1w_aligned[i] < ema20_1w_aligned[i-1]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price above midpoint or 1w trend turns up
                if close[i] > mid_20[i] or ema20_1w_aligned[i] > ema20_1w_aligned[i-1]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_Donchian_Breakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0