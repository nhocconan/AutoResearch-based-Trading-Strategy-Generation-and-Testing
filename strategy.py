#!/usr/bin/env python3
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
    
    # Get daily data for Donchian and volume (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Get weekly data for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate weekly EMA(34) for trend
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate daily Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    upper_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate daily volume average for volume filter
    vol_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to 4h timeframe
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    upper_20_aligned = align_htf_to_ltf(prices, df_1d, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1d, lower_20)
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Warmup: need all indicators
    start_idx = max(34, 20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(upper_20_aligned[i]) or 
            np.isnan(lower_20_aligned[i]) or np.isnan(vol_avg_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        ema_trend = ema_34_1w_aligned[i]
        upper_donchian = upper_20_aligned[i]
        lower_donchian = lower_20_aligned[i]
        vol_avg = vol_avg_1d_aligned[i]
        vol_current = volume[i]
        
        # Volume filter: current volume > 1.5x daily average
        volume_filter = vol_current > (vol_avg * 1.5)
        
        # Entry conditions: Donchian breakout with weekly trend and volume
        if position == 0:
            # Long: price breaks above upper Donchian + weekly trend up + volume
            if close[i] > upper_donchian and close[i] > ema_trend and volume_filter:
                signals[i] = size
                position = 1
            # Short: price breaks below lower Donchian + weekly trend down + volume
            elif close[i] < lower_donchian and close[i] < ema_trend and volume_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below lower Donchian or trend reversal
            if close[i] < lower_donchian or close[i] < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price breaks above upper Donchian or trend reversal
            if close[i] > upper_donchian or close[i] > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian20_WeeklyEMA34_Trend_VolumeFilter"
timeframe = "4h"
leverage = 1.0