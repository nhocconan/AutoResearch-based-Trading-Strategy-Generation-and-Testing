#!/usr/bin/env python3
"""
1d_WeeklyDonchian_Breakout_VolumeSpike_V1
Strategy: Daily Donchian(20) breakout with weekly trend filter and daily volume spike confirmation.
Long: Price breaks above 20-day high + weekly close > weekly SMA(10) + daily volume > 2x 20-day average volume
Short: Price breaks below 20-day low + weekly close < weekly SMA(10) + daily volume > 2x 20-day average volume
Exit: Opposite breakout or volume drop below average
Position size: 0.25
Designed to capture strong trends with volume confirmation, works in both bull and bear markets by following the weekly trend.
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
    
    # Get daily data for Donchian channels and volume
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 20-day Donchian channels on daily data
    high_max20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_min20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate weekly SMA(10) for trend filter
    sma_10_1w = pd.Series(close_1w).rolling(window=10, min_periods=10).mean().values
    
    # Calculate 20-day average volume on daily data
    volume_ma20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to daily timeframe (already aligned since we're using daily data)
    # But we need to align weekly data to daily timeframe
    sma_10_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_10_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # warmup for Donchian calculations
        # Skip if any required data is not available
        if (np.isnan(high_max20[i]) or np.isnan(low_min20[i]) or 
            np.isnan(sma_10_1w_aligned[i]) or np.isnan(volume_ma20_1d[i])):
            signals[i] = 0.0
            continue
        
        # Current daily values
        close_today = close_1d[i] if i < len(close_1d) else close[-1]
        volume_today = volume_1d[i] if i < len(volume_1d) else volume[-1]
        
        # Breakout conditions
        breakout_up = close_today > high_max20[i]
        breakout_down = close_today < low_min20[i]
        
        # Trend filter: weekly close vs weekly SMA(10)
        weekly_close = close_1w[i // 7] if i // 7 < len(close_1w) else close_1w[-1]
        weekly_sma = sma_10_1w_aligned[i]
        uptrend_weekly = weekly_close > weekly_sma
        downtrend_weekly = weekly_close < weekly_sma
        
        # Volume filter: daily volume > 2x 20-day average
        volume_filter = volume_today > (2.0 * volume_ma20_1d[i])
        
        if position == 0:
            # Long: upward breakout + weekly uptrend + volume spike
            if breakout_up and uptrend_weekly and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short: downward breakout + weekly downtrend + volume spike
            elif breakout_down and downtrend_weekly and volume_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: downward breakout or volume drop below average
            if breakout_down or volume_today < volume_ma20_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: upward breakout or volume drop below average
            if breakout_up or volume_today < volume_ma20_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyDonchian_Breakout_VolumeSpike_V1"
timeframe = "1d"
leverage = 1.0