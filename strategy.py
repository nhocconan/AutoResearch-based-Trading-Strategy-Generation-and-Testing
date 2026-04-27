#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d weekly pivot direction + volume confirmation
# Weekly pivot levels (based on Sunday close) act as strong weekly support/resistance.
# Long when price breaks above Donchian(20) high AND above weekly pivot (support).
# Short when price breaks below Donchian(20) low AND below weekly pivot (resistance).
# Volume > 1.5x 20-period average confirms breakout strength.
# Works in both bull and bear markets by requiring alignment with weekly pivot bias.
# Target: 12-30 trades/year to minimize fee decay while capturing strong weekly trend moves.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for weekly pivot calculation (need Sunday closes)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate weekly pivot: (weekly high + weekly low + weekly close) / 3
    # Using Sunday close as weekly reference (start of week)
    weekly_high = np.full(len(df_1d), np.nan)
    weekly_low = np.full(len(df_1d), np.nan)
    weekly_close = np.full(len(df_1d), np.nan)
    
    for i in range(len(df_1d)):
        # Find Sunday of current week (weekday 6)
        dt = pd.Timestamp(df_1d.index[i])
        # Go back to Sunday of this week
        days_to_sunday = (dt.weekday() + 1) % 7  # Monday=0, Sunday=6
        sunday = dt - pd.Timedelta(days=days_to_sunday)
        
        # Find index of this Sunday
        try:
            sunday_idx = df_1d.index.get_loc(sunday)
            # Get week from Sunday to Saturday
            week_end = sunday + pd.Timedelta(days=6)
            week_mask = (df_1d.index >= sunday) & (df_1d.index <= week_end)
            week_data = df_1d[week_mask]
            if len(week_data) > 0:
                weekly_high[i] = week_data['high'].max()
                weekly_low[i] = week_data['low'].min()
                weekly_close[i] = week_data['close'].iloc[-1]
        except KeyError:
            pass  # Sunday not in data
    
    # Weekly pivot point
    weekly_pivot = np.full(len(df_1d), np.nan)
    valid = ~(np.isnan(weekly_high) | np.isnan(weekly_low) | np.isnan(weekly_close))
    weekly_pivot[valid] = (weekly_high[valid] + weekly_low[valid] + weekly_close[valid]) / 3.0
    
    # Donchian channels (20-period)
    donchian_period = 20
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(donchian_period, n):
        donchian_high[i] = np.max(high[i-donchian_period:i])
        donchian_low[i] = np.min(low[i-donchian_period:i])
    
    # 20-period average volume for spike detection
    vol_ma = np.full(n, np.nan)
    vol_period = 20
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    # Align weekly pivot to 6h
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    
    signals = np.zeros(n)
    position = 0
    size = 0.25  # 25% position size
    
    # Warmup period
    start_idx = max(donchian_period, vol_period) + 50  # extra for weekly calc
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma[i]) or
            np.isnan(weekly_pivot_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Conditions:
        # 1. Donchian breakout: price breaks above/below 20-period channel
        # 2. Weekly pivot alignment: price must be on correct side of pivot
        # 3. Volume confirmation: > 1.5x average volume
        breakout_up = price > donchian_high[i]
        breakout_down = price < donchian_low[i]
        above_pivot = price > weekly_pivot_aligned[i]
        below_pivot = price < weekly_pivot_aligned[i]
        volume_confirmation = vol_ratio > 1.5
        
        if position == 0:
            # Long: breakout above Donchian high AND above weekly pivot
            if breakout_up and above_pivot and volume_confirmation:
                signals[i] = size
                position = 1
            # Short: breakout below Donchian low AND below weekly pivot
            elif breakout_down and below_pivot and volume_confirmation:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price returns to weekly pivot or breaks below Donchian low
            if price < weekly_pivot_aligned[i] or price < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: price returns to weekly pivot or breaks above Donchian high
            if price > weekly_pivot_aligned[i] or price > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Donchian_WeeklyPivot_Volume"
timeframe = "6h"
leverage = 1.0