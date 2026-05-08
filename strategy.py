#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout + weekly trend filter + volume confirmation
# Uses weekly Donchian channel to determine trend direction, 6h Donchian breakout for entry,
# and volume spike (>2x average) for confirmation. Designed to capture strong trends
# in both bull and bear markets while avoiding false breakouts in ranging conditions.
# Target: 15-35 trades/year (60-140 total over 4 years).

name = "6h_Donchian_WeeklyTrend_Volume"
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
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    # Calculate weekly Donchian channel (20-period)
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    
    highest_high_20 = np.full(len(high_weekly), np.nan)
    lowest_low_20 = np.full(len(low_weekly), np.nan)
    
    for i in range(20, len(high_weekly)):
        highest_high_20[i] = np.max(high_weekly[i-19:i+1])
        lowest_low_20[i] = np.min(low_weekly[i-19:i+1])
    
    # Get daily data for volume average
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    # Calculate daily volume average (20-period)
    vol_daily = df_daily['volume'].values
    vol_avg_20_daily = np.full(len(vol_daily), np.nan)
    
    for i in range(20, len(vol_daily)):
        vol_avg_20_daily[i] = np.mean(vol_daily[i-20:i])
    
    # Align weekly and daily indicators to 6h timeframe
    highest_high_20_aligned = align_htf_to_ltf(prices, df_weekly, highest_high_20)
    lowest_low_20_aligned = align_htf_to_ltf(prices, df_weekly, lowest_low_20)
    vol_avg_20_daily_aligned = align_htf_to_ltf(prices, df_daily, vol_avg_20_daily)
    
    # Calculate 6x Donchian channel breakout levels
    highest_high_6 = np.full(len(high), np.nan)
    lowest_low_6 = np.full(len(low), np.nan)
    
    for i in range(6, len(high)):
        highest_high_6[i] = np.max(high[i-5:i+1])
        lowest_low_6[i] = np.min(low[i-5:i+1])
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 6)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(highest_high_20_aligned[i]) or np.isnan(lowest_low_20_aligned[i]) or
            np.isnan(vol_avg_20_daily_aligned[i]) or np.isnan(highest_high_6[i]) or
            np.isnan(lowest_low_6[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume breakout: current 6h volume > 2x 20-period average of daily volume
        vol_breakout = False
        if not np.isnan(vol_avg_20_daily_aligned[i]):
            vol_breakout = volume[i] > 2.0 * vol_avg_20_daily_aligned[i]
        
        if position == 0:
            # Look for entry: follow weekly trend with 6h Donchian breakout and volume confirmation
            # Long when price breaks above 6h high AND above weekly midpoint (bullish trend)
            weekly_midpoint = (highest_high_20_aligned[i] + lowest_low_20_aligned[i]) / 2
            
            long_condition = (
                high[i] > highest_high_6[i] and     # break above 6-period high
                close[i] > weekly_midpoint and      # price above weekly midpoint (bullish bias)
                vol_breakout                        # volume confirmation
            )
            
            # Short when price breaks below 6h low AND below weekly midpoint (bearish trend)
            short_condition = (
                low[i] < lowest_low_6[i] and        # break below 6-period low
                close[i] < weekly_midpoint and      # price below weekly midpoint (bearish bias)
                vol_breakout                        # volume confirmation
            )
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns below 6h low or weekly trend turns bearish
            if low[i] < lowest_low_6[i] or close[i] < weekly_midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns above 6h high or weekly trend turns bullish
            if high[i] > highest_high_6[i] or close[i] > weekly_midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals