#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using weekly Donchian breakout with 1d ADX trend filter and volume confirmation
# Long when price breaks above weekly Donchian(20) upper band AND 1d ADX > 25 (strong trend) AND volume > 1.3 * avg_volume(20) on 6h
# Short when price breaks below weekly Donchian(20) lower band AND 1d ADX > 25 (strong trend) AND volume > 1.3 * avg_volume(20) on 6h
# Exit when price crosses back through the weekly Donchian midpoint
# Uses discrete sizing 0.25 to balance return and risk
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# Weekly Donchian provides structure that works in both bull and bear markets
# 1d ADX > 25 ensures we only trade strong trends, reducing whipsaw in ranging markets
# Volume confirmation validates breakout strength

name = "6h_WeeklyDonchian20_1dADX25_Trend_VolumeConfirm"
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
    if len(df_1w) < 20:  # Need at least 20 completed weekly bars for Donchian
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Donchian channels (20-period)
    # Upper band = highest high over 20 weeks, Lower band = lowest low over 20 weeks
    high_max_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    donchian_upper_1w = high_max_20
    donchian_lower_1w = low_min_20
    donchian_mid_1w = (donchian_upper_1w + donchian_lower_1w) / 2.0
    
    # Align weekly Donchian to 6h timeframe (wait for completed weekly bar)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1w, donchian_upper_1w)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1w, donchian_lower_1w)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1w, donchian_mid_1w)
    
    # Get 1d data ONCE before loop for ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need sufficient data for ADX calculation
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX (14-period)
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # First value has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = np.diff(high_1d, prepend=high_1d[0])
    down_move = -np.diff(low_1d, prepend=low_1d[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed values
    def WilderSmoothing(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr = WilderSmoothing(tr, 14)
    plus_di = 100 * WilderSmoothing(plus_dm, 14) / atr
    minus_di = 100 * WilderSmoothing(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = WilderSmoothing(dx, 14)
    
    # Align 1d ADX to 6h timeframe (wait for completed 1d bar)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate volume confirmation: volume > 1.3 * 20-period average volume on 6h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.3 * avg_volume_20)
    
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
                adx_aligned[i] > 25 and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly Donchian lower, 1d ADX > 25 (strong trend), volume confirmation, in session
            elif (close[i] < donchian_lower_aligned[i] and 
                  adx_aligned[i] > 25 and 
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