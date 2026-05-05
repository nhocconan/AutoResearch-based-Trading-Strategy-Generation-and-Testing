#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h Donchian(20) breakout with 1d ADX trend filter and volume confirmation
# Long when price breaks above 12h Donchian upper band AND 1d ADX > 25 (trending) AND volume > 1.8 * avg_volume(20) on 4h
# Short when price breaks below 12h Donchian lower band AND 1d ADX > 25 (trending) AND volume > 1.8 * avg_volume(20) on 4h
# Exit when price crosses back through the 12h Donchian midpoint (upper+lower)/2
# Uses discrete sizing 0.25 to balance return and risk
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# Donchian channels provide clear structure that works in both bull and bear markets
# 1d ADX filter ensures we only trade in trending regimes, reducing whipsaw in ranging markets
# Volume confirmation (1.8x) validates breakout strength with reasonable frequency

name = "4h_12hDonchian20_1dADX25_VolumeConfirm"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data ONCE before loop for Donchian calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:  # Need at least 20 completed 12h bars for Donchian
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate 12h Donchian channels (20-period)
    # Upper band = highest high over last 20 periods
    # Lower band = lowest low over last 20 periods
    high_series = pd.Series(high_12h)
    low_series = pd.Series(low_12h)
    donchian_upper_12h = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower_12h = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid_12h = (donchian_upper_12h + donchian_lower_12h) / 2.0
    
    # Align 12h Donchian to 4h timeframe (wait for completed 12h bar)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_12h, donchian_upper_12h)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_12h, donchian_lower_12h)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_12h, donchian_mid_12h)
    
    # Get 1d data ONCE before loop for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need at least 30 completed daily bars for ADX
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX (14-period)
    # ADX calculation requires +DM, -DM, and TR
    high_series_1d = pd.Series(high_1d)
    low_series_1d = pd.Series(low_1d)
    close_series_1d = pd.Series(close_1d)
    
    # True Range
    tr1 = high_series_1d - low_series_1d
    tr2 = abs(high_series_1d - close_series_1d.shift(1))
    tr3 = abs(low_series_1d - close_series_1d.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr.rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    up_move = high_series_1d.diff()
    down_move = low_series_1d.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr_1d
    minus_di = 100 * minus_dm_smooth / atr_1d
    
    # DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_1d = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align 1d ADX to 4h timeframe (wait for completed 1d bar)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate volume confirmation: volume > 1.8 * 20-period average volume on 4h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 12h Donchian upper, 1d ADX > 25 (trending), volume confirmation, in session
            if (close[i] > donchian_upper_aligned[i] and 
                adx_1d_aligned[i] > 25.0 and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 12h Donchian lower, 1d ADX > 25 (trending), volume confirmation, in session
            elif (close[i] < donchian_lower_aligned[i] and 
                  adx_1d_aligned[i] > 25.0 and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses back below 12h Donchian midpoint
            if close[i] < donchian_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses back above 12h Donchian midpoint
            if close[i] > donchian_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals