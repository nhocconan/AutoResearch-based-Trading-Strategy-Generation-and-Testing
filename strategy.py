#!/usr/bin/env python3
# Hypothesis: 1d Donchian(20) breakout with weekly trend filter and volume confirmation
# Long when price breaks above 20-day high, weekly EMA50 is rising, and volume > 1.5x 20-day average
# Short when price breaks below 20-day low, weekly EMA50 is falling, and volume > 1.5x 20-day average
# Exit when price crosses back below/above the 20-day high/low or weekly EMA trend reverses
# Position size: 0.25 (25% of capital) to balance return and drawdown
# Designed to work in trending markets via EMA filter and in ranging markets via breakout/retest

name = "1d_Donchian_WeeklyEMA_Volume_Filter"
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
    
    # Weekly EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    weekly_close = df_1w['close'].values
    weekly_ema50 = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_ema50_prev = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().shift(1).values
    weekly_ema50_rising = weekly_ema50 > weekly_ema50_prev
    weekly_ema50_falling = weekly_ema50 < weekly_ema50_prev
    
    weekly_ema50_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema50)
    weekly_ema50_rising_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema50_rising)
    weekly_ema50_falling_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema50_falling)
    
    # Donchian(20) channels from daily data
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for weekly EMA50 and Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(weekly_ema50_aligned[i]) or np.isnan(weekly_ema50_rising_aligned[i]) or 
            np.isnan(weekly_ema50_falling_aligned[i]) or np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above 20-day high, weekly EMA rising, volume spike
            if (close[i] > high_20[i] and 
                weekly_ema50_rising_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below 20-day low, weekly EMA falling, volume spike
            elif (close[i] < low_20[i] and 
                  weekly_ema50_falling_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below 20-day high OR weekly EMA turns falling
            if (close[i] < high_20[i]) or (not weekly_ema50_rising_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above 20-day low OR weekly EMA turns rising
            if (close[i] > low_20[i]) or (not weekly_ema50_falling_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals