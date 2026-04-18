#!/usr/bin/env python3
"""
1d Donchian Breakout with Weekly Trend Filter and Volume Spike
Designed for daily timeframe to capture major breakouts in direction of weekly trend.
Uses Donchian(20) breakout + volume confirmation + weekly EMA trend filter.
Aims for low trade frequency (target: 30-100 trades over 4 years) with strong edge in both bull and bear markets.
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
    
    # Get daily data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Donchian channels (20-period)
    # Upper band = highest high over last 20 days
    # Lower band = lowest low over last 20 days
    high_series = pd.Series(high_1d)
    low_series = pd.Series(low_1d)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to daily timeframe (no additional delay needed)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA34 for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume spike detection (2x 10-day average)
    vol_ma = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    entry_price = 0.0
    
    start_idx = 50  # need enough history for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper_band = donchian_upper_aligned[i]
        lower_band = donchian_lower_aligned[i]
        weekly_trend = ema_34_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian band with volume spike and above weekly EMA
            if (price > upper_band and 
                volume_spike[i] and 
                price > weekly_trend):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below lower Donchian band with volume spike and below weekly EMA
            elif (price < lower_band and 
                  volume_spike[i] and 
                  price < weekly_trend):
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long position management
            signals[i] = 0.25
            # Exit conditions: reverse signal (break below lower band)
            if price < lower_band:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position management
            signals[i] = -0.25
            # Exit conditions: reverse signal (break above upper band)
            if price > upper_band:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Donchian_Breakout_WeeklyEMA34_Volume"
timeframe = "1d"
leverage = 1.0