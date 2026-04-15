#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 250:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend (1w is HTF for 1d)
    weekly = get_htf_data(prices, '1w')
    weekly_close = weekly['close'].values
    
    # Calculate weekly EMA200 for trend filter
    weekly_close_series = pd.Series(weekly_close)
    weekly_ema200 = weekly_close_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    weekly_ema200_aligned = align_htf_to_ltf(prices, weekly, weekly_ema200)
    
    # Get daily data for price channels and volume
    daily = get_htf_data(prices, '1d')
    daily_high = daily['high'].values
    daily_low = daily['low'].values
    daily_close = daily['close'].values
    daily_volume = daily['volume'].values
    
    # Calculate daily Donchian channels (20-period)
    daily_high_series = pd.Series(daily_high)
    daily_low_series = pd.Series(daily_low)
    donchian_high = daily_high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = daily_low_series.rolling(window=20, min_periods=20).min().values
    
    # Align Donchian channels to daily timeframe (no additional delay needed for breakout)
    donchian_high_aligned = align_htf_to_ltf(prices, daily, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, daily, donchian_low)
    
    # Volume filter: daily volume > 1.5x 20-day average volume
    daily_volume_series = pd.Series(daily_volume)
    vol_ma = daily_volume_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = daily_volume > (1.5 * vol_ma)
    
    # Align volume filter to daily timeframe
    volume_filter_aligned = align_htf_to_ltf(prices, daily, volume_filter.astype(float))
    
    signals = np.zeros(n)
    
    for i in range(250, n):
        # Skip if any required data is NaN
        if (np.isnan(weekly_ema200_aligned[i]) or np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Only trade when volume filter passes
        if volume_filter_aligned[i]:
            # Long conditions: price breaks above Donchian high with volume and above weekly EMA200
            if close[i] > donchian_high_aligned[i] and close[i] > weekly_ema200_aligned[i]:
                signals[i] = 0.25
            # Short conditions: price breaks below Donchian low with volume and below weekly EMA200
            elif close[i] < donchian_low_aligned[i] and close[i] < weekly_ema200_aligned[i]:
                signals[i] = -0.25
            else:
                signals[i] = signals[i-1]
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "1d_Donchian_20_WeeklyEMA200_VolumeFilter"
timeframe = "1d"
leverage = 1.0