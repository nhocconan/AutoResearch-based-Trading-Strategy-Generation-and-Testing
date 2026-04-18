#!/usr/bin/env python3
"""
12h_1w_Donchian_20_Breakout_Volume_Trend
Hypothesis: Uses weekly Donchian channels (20-bar) as price channels. Trades breakouts of the 
weekly upper/lower band in the direction of the 12h trend (above/below 12h EMA34) with 
volume confirmation. Designed for both bull and bear markets by filtering with 12h trend. 
Target: 12-37 trades per year on 12h timeframe.
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
    
    # Get weekly data for Donchian channels
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly Donchian channels (20-period)
    lookback = 20
    upper_1w = np.full(len(high_1w), np.nan)
    lower_1w = np.full(len(low_1w), np.nan)
    
    for i in range(lookback, len(high_1w)):
        upper_1w[i] = np.max(high_1w[i-lookback:i])
        lower_1w[i] = np.min(low_1w[i-lookback:i])
    
    # Get 12h data for trend filter (EMA34)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA34
    ema_34_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= 34:
        ema_34_12h[33] = np.mean(close_12h[:34])
        multiplier = 2 / (34 + 1)
        for i in range(34, len(close_12h)):
            ema_34_12h[i] = (close_12h[i] - ema_34_12h[i-1]) * multiplier + ema_34_12h[i-1]
    
    # Align all indicators to 12h timeframe (wait for bar close)
    upper_1w_aligned = align_htf_to_ltf(prices, df_1w, upper_1w)
    lower_1w_aligned = align_htf_to_ltf(prices, df_1w, lower_1w)
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Volume confirmation: current volume > 1.5 x 28-period average (more selective)
    vol_ma = np.full(n, np.nan)
    for i in range(28, n):
        vol_ma[i] = np.mean(volume[i-28:i])
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(28, 34)  # Ensure we have enough data for volume MA and EMA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_1w_aligned[i]) or np.isnan(lower_1w_aligned[i]) or 
            np.isnan(ema_34_12h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: price breaks above weekly upper band AND above 12h EMA34, with volume
            if (close[i] > upper_1w_aligned[i] and 
                close[i] > ema_34_12h_aligned[i] and vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below weekly lower band AND below 12h EMA34, with volume
            elif (close[i] < lower_1w_aligned[i] and 
                  close[i] < ema_34_12h_aligned[i] and vol_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price returns to 12h EMA34 or breaks below weekly lower band
            if (not np.isnan(ema_34_12h_aligned[i]) and close[i] < ema_34_12h_aligned[i]) or \
               (not np.isnan(lower_1w_aligned[i]) and close[i] < lower_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to 12h EMA34 or breaks above weekly upper band
            if (not np.isnan(ema_34_12h_aligned[i]) and close[i] > ema_34_12h_aligned[i]) or \
               (not np.isnan(upper_1w_aligned[i]) and close[i] > upper_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1w_Donchian_20_Breakout_Volume_Trend"
timeframe = "12h"
leverage = 1.0