#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_Volume
Hypothesis: Camarilla pivot breakout at R1/S1 levels with 1d EMA50 trend filter and volume confirmation provides high-probability entries in trending markets while avoiding chop. Targets 20-30 trades/year on 4h timeframe. Works in both bull and bear markets by following the daily trend.
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Camarilla pivot levels for each 4h bar using previous 1d OHLC
    # We need to align the daily OHLC to each 4h bar
    # First get daily OHLC arrays
    open_1d = df_1d['open'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Align daily OHLC to 4h timeframe (each 4h bar gets the previous day's OHLC)
    open_1d_aligned = align_htf_to_ltf(prices, df_1d, open_1d)
    high_1d_aligned = align_htf_to_ltf(prices, df_1d, high_1d)
    low_1d_aligned = align_htf_to_ltf(prices, df_1d, low_1d)
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    # Calculate Camarilla levels for each bar
    # R1 = C + (H-L)*1.1/12
    # S1 = C - (H-L)*1.1/12
    rang = high_1d_aligned - low_1d_aligned
    r1 = close_1d_aligned + rang * 1.1 / 12
    s1 = close_1d_aligned - rang * 1.1 / 12
    
    # Volume confirmation: >1.5x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(r1[i]) or
            np.isnan(s1[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1d EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Breakout conditions
        breakout_long = close[i] > r1[i]  # Break above R1
        breakout_short = close[i] < s1[i]  # Break below S1
        
        # Volume confirmation
        vol_confirm = volume[i] > (1.5 * vol_ma_20[i])
        
        # Entry logic: breakout in direction of trend with volume
        long_entry = vol_confirm and uptrend and breakout_long
        short_entry = vol_confirm and downtrend and breakout_short
        
        # Exit logic: opposite breakout or trend change
        long_exit = breakout_short or (not uptrend)
        short_exit = breakout_long or (not downtrend)
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
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

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0