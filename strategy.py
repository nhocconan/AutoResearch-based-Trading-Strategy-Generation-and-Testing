#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian breakout + volume confirmation + ADX regime filter
# Long when price breaks above 4h Donchian(20) high AND volume > 1.5x 20-bar avg AND 4h ADX > 25
# Short when price breaks below 4h Donchian(20) low AND volume > 1.5x 20-bar avg AND 4h ADX > 25
# Exit when price crosses 4h Donchian midline OR volume drops below average
# Uses discrete sizing 0.20 to control fee drag and drawdown
# Target: 60-150 total trades over 4 years = 15-37/year for 1h timeframe
# 4h Donchian provides structural support/resistance, volume confirms breakout strength,
# ADX ensures we only trade in trending markets to avoid whipsaws in ranging conditions
# Session filter (08-20 UTC) reduces noise from low-liquidity periods

name = "1h_Donchian20_4hADX25_Volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data ONCE before loop for Donchian and ADX filters
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h Donchian channels (20-period)
    high_4h_series = pd.Series(high_4h)
    low_4h_series = pd.Series(low_4h)
    donchian_high = high_4h_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_4h_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Calculate 4h ADX (14-period) for trend strength filter
    # ADX requires +DI, -DI, and TR calculations
    high_4h_series = pd.Series(high_4h)
    low_4h_series = pd.Series(low_4h)
    close_4h_series = pd.Series(close_4h)
    
    # True Range
    tr1 = high_4h_series - low_4h_series
    tr2 = (high_4h_series - close_4h_series.shift(1)).abs()
    tr3 = (low_4h_series - close_4h_series.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_4h = tr.rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    up_move = high_4h_series.diff()
    down_move = low_4h_series.shift(1) - low_4h_series
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM and TR for DI calculation
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    tr_smooth = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # DI and ADX
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_4h = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align HTF indicators to 1h timeframe (wait for completed HTF bar)
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_4h, donchian_mid)
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
    
    # Volume confirmation: volume > 1.5 * 20-bar average volume (1h timeframe)
    volume_series = pd.Series(volume)
    avg_volume_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirmation = volume > (1.5 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute hours array)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(adx_4h_aligned[i]) or 
            np.isnan(volume_confirmation[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Check session filter
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Entry conditions: Donchian breakout + volume + ADX trend filter
            # Long: price breaks above Donchian high AND volume confirmation AND strong trend (ADX > 25)
            if close[i] > donchian_high_aligned[i] and volume_confirmation[i] and adx_4h_aligned[i] > 25:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below Donchian low AND volume confirmation AND strong trend (ADX > 25)
            elif close[i] < donchian_low_aligned[i] and volume_confirmation[i] and adx_4h_aligned[i] > 25:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price crosses below Donchian midline OR volume drops below average
            if close[i] < donchian_mid_aligned[i] or not volume_confirmation[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price crosses above Donchian midline OR volume drops below average
            if close[i] > donchian_mid_aligned[i] or not volume_confirmation[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals