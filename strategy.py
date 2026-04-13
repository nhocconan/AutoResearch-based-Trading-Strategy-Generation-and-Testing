#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Donchian(20) breakout with weekly pivot direction + volume confirmation
    # Long: price breaks above 6h Donchian high + weekly pivot bullish + volume > 1.5x 20-period avg
    # Short: price breaks below 6h Donchian low + weekly pivot bearish + volume > 1.5x 20-period avg
    # Exit: price returns to 6h Donchian midpoint
    # Uses weekly structure (pivots from 1w) for regime, Donchian for entries, volume for confirmation
    # Weekly pivots provide structural bias that works in both bull/bear markets
    # Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data for primary timeframe
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Get 1d data for volume (MTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    
    # Get 1w data for weekly pivot points (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 20-period Donchian channels on 6h data
    donch_high = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2
    
    # Calculate 1d volume average (20-period)
    vol_avg_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate weekly pivot points: P = (H+L+C)/3, R1 = 2*P - L, S1 = 2*P - H
    # Weekly bias: bullish if close > P, bearish if close < P
    pivot_point = (high_1w + low_1w + close_1w) / 3
    weekly_bullish = close_1w > pivot_point  # True if weekly close above pivot
    weekly_bearish = close_1w < pivot_point  # True if weekly close below pivot
    
    # Align all indicators to 6h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_6h, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_6h, donch_low)
    donch_mid_aligned = align_htf_to_ltf(prices, df_6h, donch_mid)
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):  # start from 20 to have enough data for calculations
        # Skip if data not ready
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(donch_mid_aligned[i]) or np.isnan(vol_avg_20_aligned[i]) or 
            np.isnan(weekly_bullish_aligned[i]) or np.isnan(weekly_bearish_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-period average
        curr_vol_1d = align_htf_to_ltf(prices, df_1d, volume_1d)[i]
        volume_confirmed = curr_vol_1d > 1.5 * vol_avg_20_aligned[i]
        
        # Weekly pivot direction
        is_weekly_bullish = weekly_bullish_aligned[i] > 0.5
        is_weekly_bearish = weekly_bearish_aligned[i] > 0.5
        
        # Breakout conditions with weekly pivot filter
        breakout_long = close[i] > donch_high_aligned[i] and is_weekly_bullish and volume_confirmed
        breakout_short = close[i] < donch_low_aligned[i] and is_weekly_bearish and volume_confirmed
        
        # Exit conditions: return to Donchian midpoint
        exit_long = position == 1 and close[i] <= donch_mid_aligned[i]
        exit_short = position == -1 and close[i] >= donch_mid_aligned[i]
        
        # Execute signals
        if breakout_long and position != 1:
            position = 1
            signals[i] = position_size
        elif breakout_short and position != -1:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1w_1d_donchian_breakout_volume_pivot_v1"
timeframe = "6h"
leverage = 1.0