#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-day Donchian(20) breakout with weekly ADX filter and volume confirmation
# Long when price breaks above Donchian upper band AND weekly ADX > 20 AND volume > 1.5x average
# Short when price breaks below Donchian lower band AND weekly ADX > 20 AND volume > 1.5x average
# Exit when price crosses back through the middle of the Donchian channel
# Donchian channels identify breakouts, ADX filters for trending conditions only (avoids chop),
# volume confirms institutional participation. Designed to capture trends in both bull and bear markets.
# Target: 50-100 total trades over 4 years (12-25/year) to balance opportunity and cost

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Donchian calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Donchian channels on 1d (20-period)
    highest_high = df_1d['high'].rolling(window=20, min_periods=20).max()
    lowest_low = df_1d['low'].rolling(window=20, min_periods=20).min()
    donchian_upper = highest_high.values
    donchian_lower = lowest_low.values
    donchian_middle = (donchian_upper + donchian_lower) / 2
    
    # Calculate ADX on 1w (14-period)
    df_1w = get_htf_data(prices, '1w')
    
    # True Range
    tr1 = df_1w['high'] - df_1w['low']
    tr2 = abs(df_1w['high'] - df_1w['close'].shift(1))
    tr3 = abs(df_1w['low'] - df_1w['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean()
    
    # Directional Movement
    up_move = df_1w['high'] - df_1w['high'].shift(1)
    down_move = df_1w['low'].shift(1) - df_1w['low']
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum()
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum()
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / (atr + 1e-10)
    minus_di = 100 * minus_dm_smooth / (atr + 1e-10)
    
    # ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean()
    
    # Calculate volume average for confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    # Align 1d indicators to lower timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_1d, donchian_middle)
    
    # Align 1w ADX to lower timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx.values)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 40
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_middle_aligned[i]) or
            np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        adx_val = adx_aligned[i]
        price = close[i]
        upper = donchian_upper_aligned[i]
        lower = donchian_lower_aligned[i]
        middle = donchian_middle_aligned[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.5
        
        if position == 0:
            # Long setup: price breaks above Donchian upper AND ADX > 20 (trending) AND volume confirmation
            if (price > upper and adx_val > 20 and vol > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short setup: price breaks below Donchian lower AND ADX > 20 (trending) AND volume confirmation
            elif (price < lower and adx_val > 20 and vol > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below Donchian middle
            if price < middle:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above Donchian middle
            if price > middle:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_Donchian_WeeklyADX_Volume"
timeframe = "1d"
leverage = 1.0