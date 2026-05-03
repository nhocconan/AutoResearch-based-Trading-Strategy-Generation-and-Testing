#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout + 1d weekly pivot direction + volume confirmation
# Donchian channels provide robust breakout structure in both bull and bear markets.
# Weekly pivot direction from 1d timeframe filters for higher probability trades aligned with the weekly trend.
# Volume confirmation (1.5x 20-period EMA) filters false breakouts.
# Designed for 50-150 total trades over 4 years (12-37/year) with discrete sizing to minimize fee drag.

name = "6h_Donchian20_1dWeeklyPivot_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate weekly pivot points from 1d data (using prior week's OHLC)
    # We approximate weekly by taking every 5th day (5 trading days ≈ 1 week)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly high, low, close for each point (using 5-day window)
    weekly_high = pd.Series(high_1d).rolling(window=5, min_periods=5).max().values
    weekly_low = pd.Series(low_1d).rolling(window=5, min_periods=5).min().values
    weekly_close = pd.Series(close_1d).rolling(window=5, min_periods=5).last().values
    
    # Weekly pivot point: (H + L + C) / 3
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    
    # Weekly trend: above pivot = bullish, below pivot = bearish
    weekly_trend_bullish = weekly_close > weekly_pivot
    weekly_trend_bearish = weekly_close < weekly_pivot
    
    # Align weekly pivot direction to 6h timeframe
    weekly_trend_bullish_aligned = align_htf_to_ltf(prices, df_1d, weekly_trend_bullish.astype(float))
    weekly_trend_bearish_aligned = align_htf_to_ltf(prices, df_1d, weekly_trend_bearish.astype(float))
    
    # Calculate Donchian channels from previous 6h bar (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation: 20-period EMA on 6h
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start from 20 to have valid Donchian and volume EMA
        # Skip if any value is NaN or outside session
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(vol_ema_20[i]) or np.isnan(weekly_trend_bullish_aligned[i]) or 
            np.isnan(weekly_trend_bearish_aligned[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 1.5 x 20-period EMA
        volume_spike = volume[i] > (1.5 * vol_ema_20[i])
        
        if position == 0:
            # Long: price breaks above upper Donchian in weekly bullish trend with volume spike
            if close[i] > donchian_upper[i] and weekly_trend_bullish_aligned[i] > 0.5 and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian in weekly bearish trend with volume spike
            elif close[i] < donchian_lower[i] and weekly_trend_bearish_aligned[i] > 0.5 and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below lower Donchian or weekly trend turns bearish
            if close[i] < donchian_lower[i] or weekly_trend_bearish_aligned[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above upper Donchian or weekly trend turns bullish
            if close[i] > donchian_upper[i] or weekly_trend_bullish_aligned[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals