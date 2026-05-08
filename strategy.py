#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Donchian Breakout with Weekly Trend and Volume Confirmation
# Long when price breaks above 20-day high with price above weekly EMA40 and volume > 1.5x 20-day average
# Short when price breaks below 20-day low with price below weekly EMA40 and volume > 1.5x 20-day average
# Uses daily timeframe for structure, weekly EMA for trend filter, and volume for confirmation
# Designed to work in both bull and bear markets by requiring trend alignment
# Target: 20-50 trades/year (80-200 over 4 years) to stay within optimal range

name = "1d_Donchian_Breakout_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 40:
        return np.zeros(n)
    
    weekly_close = df_weekly['close'].values
    
    # Calculate weekly EMA40 for trend filter
    weekly_close_series = pd.Series(weekly_close)
    weekly_ema40 = weekly_close_series.ewm(span=40, adjust=False, min_periods=40).mean().values
    
    # Get daily data for Donchian channels and volume
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    daily_volume = df_daily['volume'].values
    
    # Calculate 20-period Donchian channels
    donchian_high = np.full(len(daily_high), np.nan)
    donchian_low = np.full(len(daily_low), np.nan)
    
    for i in range(len(daily_high)):
        if i >= 19:
            donchian_high[i] = np.max(daily_high[i-19:i+1])
            donchian_low[i] = np.min(daily_low[i-19:i+1])
    
    # Calculate 20-day average volume for volume filter
    vol_avg_20 = np.full(len(daily_volume), np.nan)
    for i in range(len(daily_volume)):
        if i >= 19:
            vol_avg_20[i] = np.mean(daily_volume[i-19:i+1])
    
    # Align weekly EMA40 to daily timeframe
    ema40_aligned = align_htf_to_ltf(prices, df_weekly, weekly_ema40)
    
    # Align Donchian levels and volume average to daily timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_daily, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_daily, donchian_low)
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_daily, vol_avg_20)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(40, 20)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(ema40_aligned[i]) or np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or np.isnan(vol_avg_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: current volume > 1.5x 20-day average
        vol_filter = volume[i] > 1.5 * vol_avg_20_aligned[i]
        
        if position == 0:
            # Look for entry: breakout with volume and trend alignment
            # Long when price breaks above Donchian high with price above weekly EMA40
            long_condition = (
                close[i] > donchian_high_aligned[i] and 
                close[i-1] <= donchian_high_aligned[i-1] and  # just broke above
                close[i] > ema40_aligned[i] and              # bullish trend
                vol_filter
            )
            
            # Short when price breaks below Donchian low with price below weekly EMA40
            short_condition = (
                close[i] < donchian_low_aligned[i] and 
                close[i-1] >= donchian_low_aligned[i-1] and  # just broke below
                close[i] < ema40_aligned[i] and              # bearish trend
                vol_filter
            )
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price falls back below Donchian low or weekly EMA40
            if close[i] < donchian_low_aligned[i] or close[i] < ema40_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price rises back above Donchian high or weekly EMA40
            if close[i] > donchian_high_aligned[i] or close[i] > ema40_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals