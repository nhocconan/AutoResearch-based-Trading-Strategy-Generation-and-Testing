#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Weekly Donchian Breakout with Trend Filter and Volume Confirmation
# Uses weekly Donchian channels (from prior week) to identify breakout zones.
# In bullish weekly trend (price > 1d EMA34), look for long breakouts above weekly high.
# In bearish weekly trend (price < 1d EMA34), look for short breakdowns below weekly low.
# Weekly trend defined by 1d EMA34 to avoid whipsaws. Volume > 1.5x 20-period average confirms participation.
# Target: 10-25 trades/year (40-100 over 4 years) to minimize fee drag.

name = "6h_WeeklyDonchian_Trend_Volume"
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
    
    # Get weekly data for Donchian channels
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 2:
        return np.zeros(n)
    
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    
    # Get daily data for trend filter (EMA34)
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 34:
        return np.zeros(n)
    
    daily_close = df_daily['close'].values
    # Calculate EMA34 on daily data
    ema34_daily = np.full(len(daily_close), np.nan)
    if len(daily_close) >= 34:
        ema34_daily[33] = np.mean(daily_close[:34])
        for i in range(34, len(daily_close)):
            ema34_daily[i] = (daily_close[i] * 2 + ema34_daily[i-1] * 32) / 34
    
    # Calculate 20-period average volume on daily data
    daily_volume = df_daily['volume'].values
    vol_avg_20_daily = np.full(len(daily_volume), np.nan)
    if len(daily_volume) >= 20:
        for i in range(20, len(daily_volume)):
            vol_avg_20_daily[i] = np.mean(daily_volume[i-20:i])
    
    # Align weekly Donchian levels to 6h timeframe
    weekly_high_aligned = align_htf_to_ltf(prices, df_weekly, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_weekly, weekly_low)
    
    # Align daily indicators to 6h timeframe
    ema34_daily_aligned = align_htf_to_ltf(prices, df_daily, ema34_daily)
    vol_avg_20_daily_aligned = align_htf_to_ltf(prices, df_daily, vol_avg_20_daily)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(weekly_high_aligned[i]) or np.isnan(weekly_low_aligned[i]) or
            np.isnan(ema34_daily_aligned[i]) or np.isnan(vol_avg_20_daily_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: current daily volume > 1.5x 20-period average
        vol_filter = False
        if not np.isnan(vol_avg_20_daily_aligned[i]):
            # Find current daily bar's volume
            idx_daily = 0
            while idx_daily < len(df_daily) and df_daily.iloc[idx_daily]['open_time'] <= prices.iloc[i]['open_time']:
                idx_daily += 1
            idx_daily -= 1  # last completed daily bar
            
            if idx_daily >= 0:
                vol_daily_current = df_daily.iloc[idx_daily]['volume']
                vol_filter = vol_daily_current > 1.5 * vol_avg_20_daily_aligned[i]
        
        if position == 0:
            # Determine weekly trend based on price vs daily EMA34
            weekly_uptrend = close[i] > ema34_daily_aligned[i]
            weekly_downtrend = close[i] < ema34_daily_aligned[i]
            
            # Look for entry: breakout of weekly Donchian + volume
            # Long when price breaks above weekly high in uptrend with volume
            long_condition = weekly_uptrend and (close[i] > weekly_high_aligned[i]) and vol_filter
            
            # Short when price breaks below weekly low in downtrend with volume
            short_condition = weekly_downtrend and (close[i] < weekly_low_aligned[i]) and vol_filter
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns below weekly low or trend changes
            if close[i] < weekly_low_aligned[i] or close[i] < ema34_daily_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns above weekly high or trend changes
            if close[i] > weekly_high_aligned[i] or close[i] > ema34_daily_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals