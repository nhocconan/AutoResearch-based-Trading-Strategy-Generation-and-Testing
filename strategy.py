#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using weekly Donchian(20) breakout with 1w ADX25 trend filter and volume confirmation
# Long when price breaks above weekly Donchian HIGH(20) AND 1w ADX > 25 (trending) AND volume > 1.5 * avg_volume(20) on 1d
# Short when price breaks below weekly Donchian LOW(20) AND 1w ADX > 25 (trending) AND volume > 1.5 * avg_volume(20) on 1d
# Exit when price crosses back through the weekly Donchian midpoint (HIGH+LOW)/2
# Uses discrete sizing 0.30 to balance return and risk
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe
# Weekly Donchian(20) provides strong breakout levels that reduce whipsaw
# 1w ADX25 trend filter ensures we trade only in trending markets (avoids chop)
# Volume confirmation (1.5x) validates breakout strength while limiting overtrading
# Works in both bull and bear markets by capturing strong directional moves

name = "1d_WeeklyDonchian20_1wADX25_Trend_VolumeConfirm"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data ONCE before loop for Donchian and ADX calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:  # Need at least 20 completed weekly bars for Donchian
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Donchian channels (20-period)
    # Donchian HIGH = max(high, lookback=20), LOW = min(low, lookback=20)
    high_20_1w = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    low_20_1w = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    donchian_high_1w = high_20_1w
    donchian_low_1w = low_20_1w
    donchian_mid_1w = (donchian_high_1w + donchian_low_1w) / 2.0
    
    # Calculate 1w ADX(14) for trend filter
    # ADX requires +DI, -DI, and TR calculations
    high_1w_series = pd.Series(high_1w)
    low_1w_series = pd.Series(low_1w)
    close_1w_series = pd.Series(close_1w)
    
    # True Range
    tr1 = high_1w_series - low_1w_series
    tr2 = abs(high_1w_series - close_1w_series.shift(1))
    tr3 = abs(low_1w_series - close_1w_series.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1w = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Movement
    up_move = high_1w_series - high_1w_series.shift(1)
    down_move = low_1w_series.shift(1) - low_1w_series
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM and DI
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_di_1w = 100 * plus_dm_smooth / atr_1w
    minus_di_1w = 100 * minus_dm_smooth / atr_1w
    
    # DX and ADX
    dx = 100 * abs(plus_di_1w - minus_di_1w) / (plus_di_1w + minus_di_1w)
    adx_1w = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align weekly indicators to 1d timeframe (wait for completed weekly bar)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high_1w)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low_1w)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1w, donchian_mid_1w)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 1d
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(adx_1w_aligned[i]) or 
            np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above weekly Donchian HIGH, ADX > 25 (trending), volume confirmation, in session
            if (close[i] > donchian_high_aligned[i] and 
                adx_1w_aligned[i] > 25.0 and 
                volume_confirm[i]):
                signals[i] = 0.30
                position = 1
            # Short: price breaks below weekly Donchian LOW, ADX > 25 (trending), volume confirmation, in session
            elif (close[i] < donchian_low_aligned[i] and 
                  adx_1w_aligned[i] > 25.0 and 
                  volume_confirm[i]):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price crosses back below weekly Donchian midpoint
            if close[i] < donchian_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price crosses back above weekly Donchian midpoint
            if close[i] > donchian_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals