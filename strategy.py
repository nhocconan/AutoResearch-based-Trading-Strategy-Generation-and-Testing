#!/usr/bin/env python3
"""
12h_Donchian_20_Trend_Filter_Volume_Confirmation
Hypothesis: 12h timeframe with daily Donchian(20) breakout for entry and weekly EMA40 trend filter.
Enters long when price breaks above Donchian upper band in weekly uptrend with volume confirmation.
Enters short when price breaks below Donchian lower band in weekly downtrend with volume confirmation.
Exits when price crosses opposite Donchian band or trend reverses.
Designed for 12-30 trades/year to minimize fee drag in 12h timeframe.
Uses volume > 1.5x 20-period average for confirmation to avoid false breaks.
"""

name = "12h_Donchian_20_Trend_Filter_Volume_Confirmation"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 40:
        return np.zeros(n)
    
    # Calculate weekly EMA40 for trend
    close_1w = df_1w['close'].values
    ema_40_1w = pd.Series(close_1w).ewm(span=40, adjust=False, min_periods=40).mean().values
    ema_40_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_40_1w)
    
    # Get daily data for Donchian channels (20-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily Donchian channels (20-period high/low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Donchian upper = 20-period high, lower = 20-period low
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Get 12h volume for confirmation
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.divide(volume, vol_ma20, out=np.zeros_like(volume), where=vol_ma20!=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Warmup
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_40_1w_aligned[i]) or 
            np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine weekly trend
        # Get weekly close aligned to 12h for trend comparison
        close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
        if np.isnan(close_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        trend_up = close_1w_aligned[i] > ema_40_1w_aligned[i]
        trend_down = close_1w_aligned[i] < ema_40_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper in weekly uptrend with volume
            if (high[i] > donchian_high_aligned[i] and 
                close[i] > donchian_high_aligned[i] and  # close confirmation
                vol_ratio[i] > 1.5 and 
                trend_up):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower in weekly downtrend with volume
            elif (low[i] < donchian_low_aligned[i] and 
                  close[i] < donchian_low_aligned[i] and  # close confirmation
                  vol_ratio[i] > 1.5 and 
                  trend_down):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below Donchian lower or trend turns down
            if (low[i] < donchian_low_aligned[i] and 
                close[i] < donchian_low_aligned[i]) or \
               not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above Donchian upper or trend turns up
            if (high[i] > donchian_high_aligned[i] and 
                close[i] > donchian_high_aligned[i]) or \
               not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals