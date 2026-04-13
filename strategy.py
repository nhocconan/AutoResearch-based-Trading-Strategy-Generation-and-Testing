#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Donchian(20) breakout + 1w pivot direction + 1d volume confirmation
    # Long: price breaks above 20-period high + 1w close > 1w pivot + 1d volume > 1.5x 20-period average
    # Short: price breaks below 20-period low + 1w close < 1w pivot + 1d volume > 1.5x 20-period average
    # Uses discrete sizing (0.25) to minimize fee drag and ATR-based stoploss
    # Target: 12-37 trades/year to stay within 6h optimal range
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Get 1w data for pivot and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 6h Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1w pivot (based on previous week)
    # Pivot = (H+L+C)/3
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    
    # Calculate 1d volume average (20-period) for confirmation
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 6h timeframe
    high_roll_aligned = align_htf_to_ltf(prices, prices, high_roll)  # 6h data
    low_roll_aligned = align_htf_to_ltf(prices, prices, low_roll)    # 6h data
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    # Track entry price for stoploss
    entry_price = np.full(n, np.nan)
    atr_6h = np.zeros(n)  # ATR using 6h range
    
    # Calculate ATR (6h range approximation)
    for i in range(n):
        daily_range = high[i] - low[i]
        atr_6h[i] = daily_range * 0.5  # Approximate ATR as 50% of 6h range
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(high_roll_aligned[i]) or 
            np.isnan(low_roll_aligned[i]) or
            np.isnan(pivot_1w_aligned[i]) or
            np.isnan(vol_avg_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-period average
        idx_1d = i // 4  # 6h bars in 1d timeframe (4 bars per day)
        if idx_1d >= len(volume_1d):
            signals[i] = 0.0
            continue
        volume_confirmed = volume_1d[idx_1d] > 1.5 * vol_avg_20_1d_aligned[i]
        
        # Trend filter: 1w close above/below pivot
        idx_1w = i // 28  # 6h bars in 1w timeframe (28 bars per week)
        if idx_1w >= len(close_1w):
            signals[i] = 0.0
            continue
        uptrend = close_1w[idx_1w] > pivot_1w_aligned[i]
        downtrend = close_1w[idx_1w] < pivot_1w_aligned[i]
        
        # Breakout conditions: price breaks Donchian levels with volume and trend
        breakout_long = (close[i] > high_roll_aligned[i]) and volume_confirmed and uptrend
        breakout_short = (close[i] < low_roll_aligned[i]) and volume_confirmed and downtrend
        
        # Stoploss: 2.0x ATR below/above entry
        exit_long = position == 1 and not np.isnan(entry_price[i-1]) and close[i] < entry_price[i-1] - 2.0 * atr_6h[i]
        exit_short = position == -1 and not np.isnan(entry_price[i-1]) and close[i] > entry_price[i-1] + 2.0 * atr_6h[i]
        
        # Execute signals
        if breakout_long and position != 1:
            position = 1
            signals[i] = position_size
            entry_price[i] = close[i]
        elif breakout_short and position != -1:
            position = -1
            signals[i] = -position_size
            entry_price[i] = close[i]
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
            entry_price[i] = np.nan
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
            entry_price[i] = np.nan
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
                entry_price[i] = entry_price[i-1] if i > 0 else np.nan
            elif position == -1:
                signals[i] = -position_size
                entry_price[i] = entry_price[i-1] if i > 0 else np.nan
            else:
                signals[i] = 0.0
                entry_price[i] = np.nan
    
    return signals

name = "6h_1d_1w_donchian_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0