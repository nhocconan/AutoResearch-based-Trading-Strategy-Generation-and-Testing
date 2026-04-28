#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using weekly Donchian channel breakouts with 1d ADX trend filter and volume confirmation.
# Enter long when price breaks above weekly Donchian(20) upper band with 1d ADX > 25 and volume > 1.5x 20-bar average.
# Enter short when price breaks below weekly Donchian(20) lower band with 1d ADX > 25 and volume > 1.5x 20-bar average.
# Exit when price returns to weekly Donchian midpoint.
# Uses discrete position sizing (0.25) to control risk and minimize fee churn.
# Weekly Donchian provides stable structure, 1d ADX filters for trending markets only, volume confirms breakout strength.
# Designed to work in both bull (breakouts with trend) and bear (breakdowns with trend) markets by requiring ADX > 25.
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

name = "6h_WeeklyDonchian20_1dADX25_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian channel calculation (MTF structure)
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly Donchian(20) channels
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Donchian upper: max(high, lookback=20)
    roll_max = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    # Donchian lower: min(low, lookback=20)
    roll_min = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    # Donchian midpoint: (upper + lower) / 2
    donchian_mid = (roll_max + roll_min) / 2.0
    
    # Align weekly Donchian levels to 6h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_1w, roll_max)
    lower_aligned = align_htf_to_ltf(prices, df_1w, roll_min)
    mid_aligned = align_htf_to_ltf(prices, df_1w, donchian_mid)
    
    # Get daily data for ADX trend filter (MTF trend)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:  # Need sufficient data for ADX calculation
        return np.zeros(n)
    
    # Calculate 1d ADX(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - close_1d)
    tr3 = np.abs(low_1d - close_1d)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(np.concatenate([[0.0], plus_dm])).rolling(window=14, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(np.concatenate([[0.0], minus_dm])).rolling(window=14, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr
    minus_di = 100 * minus_dm_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align 1d ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or np.isnan(mid_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # Trend filter: 1d ADX > 25 (trending market)
        trending = adx_aligned[i] > 25.0
        
        # Weekly Donchian breakout conditions
        long_breakout = close[i] > upper_aligned[i]
        short_breakout = close[i] < lower_aligned[i]
        
        # Exit condition: return to weekly Donchian midpoint
        long_exit = close[i] < mid_aligned[i]
        short_exit = close[i] > mid_aligned[i]
        
        # Entry conditions
        long_entry = long_breakout and vol_confirm and trending
        short_entry = short_breakout and vol_confirm and trending
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals