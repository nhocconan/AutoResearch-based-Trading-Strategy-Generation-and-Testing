#!/usr/bin/env python3

"""
Hypothesis: 4-hour Donchian channel breakout with 12-hour trend filter and volume confirmation.
Only trade long when price breaks above 20-period Donchian upper channel during 12-hour uptrend with volume spike.
Short when price breaks below 20-period Donchian lower channel during 12-hour downtrend with volume spike.
Exit when price returns to the Donchian middle or trend reverses.
Designed for moderate trade frequency (20-50 trades/year) by requiring trend alignment and volume confirmation.
Works in both bull and bear markets by following the 12-hour trend.
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
    
    # Donchian Channel (20-period) on 4h
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2
    
    # Load 12-hour data for trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # 12-period EMA on 12h close for trend
    close_12h = df_12h['close'].values
    ema12_12h = pd.Series(close_12h).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema12_12h_aligned = align_htf_to_ltf(prices, df_12h, ema12_12h)
    
    # Volume confirmation: current volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(40, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema12_12h_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 1.8 * vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian + 12h uptrend + volume spike
            if close[i] > donchian_upper[i] and ema12_12h_aligned[i] > ema12_12h_aligned[i-1] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian + 12h downtrend + volume spike
            elif close[i] < donchian_lower[i] and ema12_12h_aligned[i] < ema12_12h_aligned[i-1] and vol_spike:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to middle or trend reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: price below middle or 12h trend turns down
                if close[i] < donchian_middle[i] or ema12_12h_aligned[i] < ema12_12h_aligned[i-1]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price above middle or 12h trend turns up
                if close[i] > donchian_middle[i] or ema12_12h_aligned[i] > ema12_12h_aligned[i-1]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian_Breakout_12hTrend_Volume"
timeframe = "4h"
leverage = 1.0