# 6h Donchian(20) breakout with weekly pivot direction and volume confirmation
# Uses 6h Donchian breakouts filtered by weekly pivot trend (bullish/bearish)
# and volume spikes to capture strong momentum moves.
# Designed to work in both bull and bear markets by using weekly pivot
# as trend filter and requiring volume confirmation to avoid false breakouts.
# Target: 50-150 total trades over 4 years (~12-37/year) to minimize fee drag.

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
    
    # Get daily data for weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot from daily data (using last 5 days)
    # Weekly pivot = (weekly high + weekly low + weekly close) / 3
    weekly_high = np.full(n, np.nan)
    weekly_low = np.full(n, np.nan)
    weekly_close = np.full(n, np.nan)
    
    for i in range(len(prices)):
        # Get index of daily data up to current 6h bar
        # Since we're using 6h timeframe, we need to map to daily bars
        # Simplified: use previous day's data for weekly calculation
        pass
    
    # Simpler approach: calculate weekly pivot on daily data and align
    # For weekly pivot, we need weekly high/low/close
    # We'll approximate using daily data: weekly high = max of last 5 daily highs
    # This is a simplification but should work for demonstration
    
    # Calculate rolling weekly high/low/close from daily data
    # We'll use a simpler method: calculate pivot points from previous week
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot using last 5 daily bars (approximation)
    weekly_pivot = np.full(len(high_1d), np.nan)
    for i in range(4, len(high_1d)):  # Need at least 5 days
        week_high = np.max(high_1d[i-4:i+1])
        week_low = np.min(low_1d[i-4:i+1])
        week_close = close_1d[i]
        weekly_pivot[i] = (week_high + week_low + week_close) / 3
    
    # Align weekly pivot to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    
    # Get 6h data for Donchian channels
    # Calculate 20-period Donchian channels directly on 6h data
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(lookback, n):
        highest_high[i] = np.max(high[i-lookback:i])
        lowest_low[i] = np.min(low[i-lookback:i])
    
    # 20-period average volume for spike detection
    vol_ma = np.full(n, np.nan)
    vol_period = 20
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25  # 25% position size
    
    # Warmup period
    start_idx = max(lookback, vol_period, 5)  # Need at least 5 days for weekly pivot
    
    for i in range(start_idx, n):
        if (np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i]) or
            np.isnan(weekly_pivot_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Determine trend from weekly pivot
        # Bullish if price above weekly pivot, bearish if below
        bullish = price > weekly_pivot_aligned[i]
        bearish = price < weekly_pivot_aligned[i]
        
        # Volume confirmation: spike > 2.0x average
        volume_confirmation = vol_ratio > 2.0
        
        if position == 0:
            # Long breakout: price breaks above Donchian high in bullish trend with volume
            if bullish and price > highest_high[i] and volume_confirmation:
                signals[i] = size
                position = 1
            # Short breakdown: price breaks below Donchian low in bearish trend with volume
            elif bearish and price < lowest_low[i] and volume_confirmation:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price breaks below Donchian low or trend turns bearish
            if price < lowest_low[i] or bearish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: price breaks above Donchian high or trend turns bullish
            if price > highest_high[i] or bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Donchian20_WeeklyPivot_Trend_Volume"
timeframe = "6h"
leverage = 1.0