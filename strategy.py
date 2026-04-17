#!/usr/bin/env python3
"""
Hypothesis: 1h strategy using 4h Donchian breakout with volume confirmation and 1d EMA50 trend filter.
Long when price breaks above 4h Donchian upper (20) AND 1h volume > 1.5x 20-bar average AND close > 1d EMA50.
Short when price breaks below 4h Donchian lower (20) AND 1h volume > 1.5x 20-bar average AND close < 1d EMA50.
Exit when price touches the 4h Donchian midpoint or opposite band.
Uses 4h for structure/direction, 1h for entry timing precision, 1d for trend filter.
Target: 15-35 trades/year per symbol (60-140 total over 4 years) to minimize fee drag.
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
    
    # Get 4h data for Donchian channels (structure/direction)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h Donchian channels (20-period)
    donchian_window = 20
    upper_4h = pd.Series(high_4h).rolling(window=donchian_window, min_periods=donchian_window).max().values
    lower_4h = pd.Series(low_4h).rolling(window=donchian_window, min_periods=donchian_window).min().values
    mid_4h = (upper_4h + lower_4h) / 2
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1h volume MA for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align all HTF indicators to 1h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_4h, upper_4h)
    lower_aligned = align_htf_to_ltf(prices, df_4h, lower_4h)
    mid_aligned = align_htf_to_ltf(prices, df_4h, mid_4h)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup period
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_aligned[i]) or 
            np.isnan(lower_aligned[i]) or
            np.isnan(mid_aligned[i]) or
            np.isnan(ema50_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1h volume > 1.5x 20-bar average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # Breakout conditions
        breakout_upper = close[i] > upper_aligned[i]
        breakout_lower = close[i] < lower_aligned[i]
        
        # Exit conditions: touch midpoint or opposite band
        touch_mid = abs(close[i] - mid_aligned[i]) < 0.001 * close[i]  # within 0.1%
        touch_opposite = (position == 1 and close[i] < lower_aligned[i]) or \
                         (position == -1 and close[i] > upper_aligned[i])
        
        if position == 0:
            # Long: break above upper with volume confirmation and uptrend (close > EMA50)
            if (breakout_upper and volume_confirmed and close[i] > ema50_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short: break below lower with volume confirmation and downtrend (close < EMA50)
            elif (breakout_lower and volume_confirmed and close[i] < ema50_aligned[i]):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: touch midpoint or break below lower
            if (touch_mid or touch_opposite):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: touch midpoint or break above upper
            if (touch_mid or touch_opposite):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Donchian20_Volume_EMA50_Trend"
timeframe = "1h"
leverage = 1.0