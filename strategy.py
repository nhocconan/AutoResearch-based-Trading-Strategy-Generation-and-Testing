#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_Volume_Trend_Filter_v1
4h strategy using Donchian channel breakout with volume confirmation and EMA trend filter.
Enters long when price breaks above Donchian upper band with volume above average and EMA(50) rising.
Enters short when price breaks below Donchian lower band with volume above average and EMA(50) falling.
Exits when price returns to the Donchian middle (mean) or trend filter fails.
Uses 1d EMA(50) as higher timeframe trend filter to avoid counter-trend trades.
Target: 100-200 total trades over 4 years (25-50/year).
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
    
    # === Calculate Donchian Channel (20) on 4h ===
    # Upper band: 20-period high
    upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Lower band: 20-period low
    lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    # Middle band: average of upper and lower
    middle = (upper + lower) / 2
    
    # === Calculate 1d EMA(50) for Trend Filter ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    # Trend: 1 if EMA rising, -1 if falling
    ema_trend = np.where(ema_50_1d_aligned > np.roll(ema_50_1d_aligned, 1), 1, -1)
    ema_trend[0] = 1  # initialize
    
    # === 4h Volume for Confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(upper[i]) or 
            np.isnan(lower[i]) or 
            np.isnan(middle[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # Breakout conditions
        breakout_long = close[i] > upper[i]
        breakout_short = close[i] < lower[i]
        
        # Return to middle (exit condition)
        return_to_middle = abs(close[i] - middle[i]) < 0.001 * middle[i]
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: break above upper with volume and uptrend
            if breakout_long and vol_confirmed and ema_trend[i] == 1:
                signals[i] = 0.25
                position = 1
                continue
            # Short: break below lower with volume and downtrend
            elif breakout_short and vol_confirmed and ema_trend[i] == -1:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price returns to middle OR trend turns down
            if return_to_middle or ema_trend[i] == -1:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to middle OR trend turns up
            if return_to_middle or ema_trend[i] == 1:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_Volume_Trend_Filter_v1"
timeframe = "4h"
leverage = 1.0