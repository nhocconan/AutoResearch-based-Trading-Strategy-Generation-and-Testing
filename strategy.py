#!/usr/bin/env python3
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
    
    # Get weekly data for trend direction
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA34 for trend
    close_1w_series = pd.Series(close_1w)
    ema34_1w = close_1w_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align weekly EMA to 6h timeframe
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Get daily data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate daily Donchian channels (20-period)
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align daily Donchian to 6h timeframe
    donchian_high_6h = align_htf_to_ltf(prices, df_1d, high_20)
    donchian_low_6h = align_htf_to_ltf(prices, df_1d, low_20)
    
    # Volume filter: current volume > 1.5 * 50-period average
    volume_ma50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema34_1w_aligned[i]) or np.isnan(donchian_high_6h[i]) or 
            np.isnan(donchian_low_6h[i]) or np.isnan(volume_ma50[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > (1.5 * volume_ma50[i])
        
        if position == 0:
            # Long: price breaks above daily Donchian high AND weekly EMA34 is rising
            if close[i] > donchian_high_6h[i] and ema34_1w_aligned[i] > ema34_1w_aligned[i-1] and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below daily Donchian low AND weekly EMA34 is falling
            elif close[i] < donchian_low_6h[i] and ema34_1w_aligned[i] < ema34_1w_aligned[i-1] and volume_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls below daily Donchian low
            if close[i] < donchian_low_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises above daily Donchian high
            if close[i] > donchian_high_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyEMA34_DailyDonchian_Breakout_Volume"
timeframe = "6h"
leverage = 1.0