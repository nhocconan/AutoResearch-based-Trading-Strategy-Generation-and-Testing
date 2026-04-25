#!/usr/bin/env python3
"""
6h Donchian(20) breakout + 1d Weekly Pivot Direction + Volume Spike Confirmation
Hypothesis: Donchian channel breakouts capture strong momentum. Weekly pivot direction (from 1d data) provides
higher-timeframe bias: only take long breaks above Donchian high when weekly pivot is bullish (price above
weekly pivot point), and short breaks below Donchian low when bearish (price below weekly pivot).
Volume spike (>2.0x 20-bar volume MA) confirms momentum. Works in bull markets via upside breakouts with
bullish weekly bias and in bear markets via downside breakouts with bearish weekly bias. Targets 12-37 trades/year
to avoid fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for weekly pivot calculation (using prior week's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:  # Need sufficient data for weekly aggregation
        return np.zeros(n)
    
    # Calculate weekly OHLC from daily data
    # Group by week starting Monday
    df_1d = df_1d.copy()
    df_1d['week_start'] = pd.to_datetime(df_1d.index).to_period('W').start_time
    weekly = df_1d.groupby('week_start').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).reset_index()
    
    if len(weekly) < 2:
        return np.zeros(n)
    
    # Weekly pivot point: (Prior week HIGH + LOW + CLOSE) / 3
    weekly_high = weekly['high'].shift(1).values
    weekly_low = weekly['low'].shift(1).values
    weekly_close = weekly['close'].shift(1).values
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    
    # Align weekly pivot to 1d timeframe (each daily bar gets prior week's pivot)
    weekly_pivot_1d = np.repeat(weekly_pivot, 7)  # Approximate: each weekly pivot applies to ~7 days
    # Trim to match df_1d length
    if len(weekly_pivot_1d) > len(df_1d):
        weekly_pivot_1d = weekly_pivot_1d[:len(df_1d)]
    elif len(weekly_pivot_1d) < len(df_1d):
        # Pad with last known value
        padding = np.full(len(df_1d) - len(weekly_pivot_1d), weekly_pivot_1d[-1] if len(weekly_pivot_1d) > 0 else np.nan)
        weekly_pivot_1d = np.concatenate([weekly_pivot_1d, padding])
    
    # Align weekly pivot to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot_1d)
    
    # Get 6h data for Donchian channel (20-period)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 21:  # Need 20 for Donchian + 1
        return np.zeros(n)
    
    # Calculate 20-period Donchian high/low on 6h data
    high_6h = pd.Series(df_6h['high'])
    low_6h = pd.Series(df_6h['low'])
    donchian_high = high_6h.rolling(window=20, min_periods=20).max().values
    donchian_low = low_6h.rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to primary timeframe (6h)
    donchian_high_aligned = align_htf_to_ltf(prices, df_6h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_6h, donchian_low)
    
    # Calculate 20-period volume MA for volume spike confirmation (6h)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Donchian, weekly pivot, and volume MA
    start_idx = max(21, 20)  # 21 for Donchian (20 + 1), 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        dh_val = donchian_high_aligned[i]
        dl_val = donchian_low_aligned[i]
        wp_val = weekly_pivot_aligned[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        # Weekly pivot bias: bullish if price above weekly pivot, bearish if below
        weekly_bullish = curr_close > wp_val
        weekly_bearish = curr_close < wp_val
        
        if position == 0:
            # Long: break above Donchian high + weekly bullish bias + volume confirmation
            long_signal = (curr_high > dh_val) and weekly_bullish and volume_confirm
            # Short: break below Donchian low + weekly bearish bias + volume confirmation
            short_signal = (curr_low < dl_val) and weekly_bearish and volume_confirm
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses back below Donchian high OR weekly bias turns bearish
            if (curr_close < dh_val) or (not weekly_bullish):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses back above Donchian low OR weekly bias turns bullish
            if (curr_close > dl_val) or (not weekly_bearish):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_WeeklyPivot_Direction_VolumeConfirm"
timeframe = "6h"
leverage = 1.0