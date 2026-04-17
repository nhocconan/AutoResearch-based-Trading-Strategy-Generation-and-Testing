#!/usr/bin/env python3
"""
Hypothesis: 1d timeframe with 1w EMA200 trend filter + 1w Donchian(20) breakout + volume confirmation.
Long when price breaks above weekly Donchian upper band with 1w close > EMA200 and volume > 2x 20-period 1d volume average.
Short when price breaks below weekly Donchian lower band with 1w close < EMA200 and volume > 2x 20-period 1d volume average.
Uses discrete position sizing 0.30 to limit fee drag. Target: 30-100 total trades over 4 years.
Weekly Donchian provides structural breakout levels; EMA200 filters for major trend direction only; volume confirms institutional participation.
Designed to work in bull markets (breakout continuation with trend) and bear markets (strong trend continuation breakdowns).
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
    
    # Get 1d data for volume
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # Get 1w data for EMA200 and Donchian
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA200
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate weekly Donchian channels (20-period)
    upper_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d volume 20-period average
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all to 1d
    ema200_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    upper_donchian_aligned = align_htf_to_ltf(prices, df_1w, upper_20)
    lower_donchian_aligned = align_htf_to_ltf(prices, df_1w, lower_20)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 200  # need enough for EMA200 and Donchian
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema200_aligned[i]) or np.isnan(upper_donchian_aligned[i]) or 
            np.isnan(lower_donchian_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i]) or 
            np.isnan(volume_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 2x 20-period average
        volume_confirmed = volume_1d_aligned[i] > 2.0 * vol_ma_20_1d_aligned[i]
        # Trend filter: price relative to EMA200
        above_ema200 = close_1w[-1] > ema200_1w[-1] if len(close_1w) > 0 else False  # Use latest weekly close
        below_ema200 = close_1w[-1] < ema200_1w[-1] if len(close_1w) > 0 else False
        
        if position == 0:
            # Long: price breaks above weekly Donchian upper band with above EMA200 and volume
            if (close[i] > upper_donchian_aligned[i] and 
                above_ema200 and 
                volume_confirmed):
                signals[i] = 0.30
                position = 1
            # Short: price breaks below weekly Donchian lower band with below EMA200 and volume
            elif (close[i] < lower_donchian_aligned[i] and 
                  below_ema200 and 
                  volume_confirmed):
                signals[i] = -0.30
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below weekly Donchian middle (mean of upper/lower)
            middle = (upper_donchian_aligned[i] + lower_donchian_aligned[i]) / 2.0
            if close[i] < middle:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:
            # Exit short: price rises back above weekly Donchian middle
            middle = (upper_donchian_aligned[i] + lower_donchian_aligned[i]) / 2.0
            if close[i] > middle:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "1d_1wEMA200_Donchian20_Breakout_Volume_Confirm"
timeframe = "1d"
leverage = 1.0