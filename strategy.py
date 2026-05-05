#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using weekly Donchian(20) breakout with 1d ADX25 trend filter and volume confirmation
# Long when price breaks above weekly Donchian upper band (20-period high) AND 1d ADX > 25 (strong trend) AND volume > 1.5 * avg_volume(20) on 6h
# Short when price breaks below weekly Donchian lower band (20-period low) AND 1d ADX > 25 (strong trend) AND volume > 1.5 * avg_volume(20) on 6h
# Exit when price crosses back through the weekly Donchian midpoint (upper+lower)/2
# Uses discrete sizing 0.25 to balance return and risk
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# Weekly Donchian provides strong structural breakout levels that reduce whipsaw
# 1d ADX25 trend filter ensures we only trade in strong trending markets (avoids ranging chop)
# Volume confirmation (1.5x) validates breakout strength while limiting overtrading
# This combination has not been extensively tried in the 6h timeframe with weekly structure

name = "6h_WeeklyDonchian20_1dADX25_VolumeConfirm"
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
    
    # Get weekly data ONCE before loop for Donchian calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:  # Need at least 20 completed weekly bars for Donchian20
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly Donchian bands (20-period)
    # Upper band = highest high over last 20 weekly periods
    # Lower band = lowest low over last 20 weekly periods
    high_series_1w = pd.Series(high_1w)
    low_series_1w = pd.Series(low_1w)
    donchian_upper_1w = high_series_1w.rolling(window=20, min_periods=20).max().values
    donchian_lower_1w = low_series_1w.rolling(window=20, min_periods=20).min().values
    donchian_mid_1w = (donchian_upper_1w + donchian_lower_1w) / 2.0
    
    # Align weekly Donchian to 6h timeframe (wait for completed weekly bar)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1w, donchian_upper_1w)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1w, donchian_lower_1w)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1w, donchian_mid_1w)
    
    # Get 1d data ONCE before loop for ADX25 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need at least 30 completed daily bars for ADX calculation
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX (14-period)
    # ADX calculation: +DM, -DM, TR, then smoothed, then DX, then ADX
    high_series_1d = pd.Series(high_1d)
    low_series_1d = pd.Series(low_1d)
    close_series_1d = pd.Series(close_1d)
    
    # True Range
    tr1 = high_series_1d - low_series_1d
    tr2 = abs(high_series_1d - close_series_1d.shift(1))
    tr3 = abs(low_series_1d - close_series_1d.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    
    # Directional Movement
    up_move = high_series_1d - high_series_1d.shift(1)
    down_move = low_series_1d.shift(1) - low_series_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr
    minus_di = 100 * minus_dm_smooth / atr
    
    # DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    adx_values = adx.values
    
    # Align 1d ADX to 6h timeframe (wait for completed daily bar)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_values)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 6h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above weekly Donchian upper, 1d ADX > 25 (strong trend), volume confirmation, in session
            if (close[i] > donchian_upper_aligned[i] and 
                adx_aligned[i] > 25.0 and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly Donchian lower, 1d ADX > 25 (strong trend), volume confirmation, in session
            elif (close[i] < donchian_lower_aligned[i] and 
                  adx_aligned[i] > 25.0 and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses back below weekly Donchian midpoint
            if close[i] < donchian_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses back above weekly Donchian midpoint
            if close[i] > donchian_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals