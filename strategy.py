#!/usr/bin/env python3
"""
6h_Donchian_20_Breakout_1dTrend_Volume
Hypothesis: Uses 6h Donchian channel breakout (20-period) with 1d EMA100 trend filter and volume confirmation (2x 96-bar avg) to capture high-probability breakouts. Designed for low trade frequency (12-37/year) to minimize fee flood. Works in both bull and bear by following 1d trend direction. Volume confirmation avoids fakeouts. Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    
    # Calculate 1d EMA100 for trend filter
    close_1d = df_1d['close'].values
    ema_100_1d = pd.Series(close_1d).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema_100_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_100_1d)
    
    # Volume confirmation: >2x 96-period MA (4 days of 6h bars)
    vol_ma_96 = pd.Series(volume).rolling(window=96, min_periods=96).mean().values
    
    # Calculate Donchian channel (20-period) on 6h data
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for EMA100 and Donchian to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_100_1d_aligned[i]) or 
            np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i]) or
            np.isnan(vol_ma_96[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1d EMA100
        uptrend = close[i] > ema_100_1d_aligned[i]
        downtrend = close[i] < ema_100_1d_aligned[i]
        
        # Volume confirmation (>2x average)
        vol_confirm = volume[i] > (2.0 * vol_ma_96[i])
        
        # Breakout conditions at Donchian levels
        long_breakout = close[i] > donchian_high[i] and vol_confirm and uptrend
        short_breakout = close[i] < donchian_low[i] and vol_confirm and downtrend
        
        # Exit conditions: return to midpoint of Donchian channel
        midpoint = (donchian_high[i] + donchian_low[i]) / 2
        long_exit = close[i] < midpoint
        short_exit = close[i] > midpoint
        
        if long_breakout and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_breakout and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_Donchian_20_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0