#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian(20) breakout with 1d ADX25 trend filter and volume confirmation
# Long when price breaks above 4h Donchian upper AND 1d ADX > 25 (trending) AND volume > 1.5 * avg_volume(20) on 1h
# Short when price breaks below 4h Donchian lower AND 1d ADX > 25 (trending) AND volume > 1.5 * avg_volume(20) on 1h
# Exit when price crosses back through the 4h Donchian midpoint (upper+lower)/2
# Uses discrete sizing 0.20 to balance return and risk
# Target: 60-150 total trades over 4 years = 15-37/year for 1h timeframe
# 4h Donchian provides structure that works in both bull and bear markets
# 1d ADX filter ensures we trade with the higher timeframe trend, reducing whipsaw
# Volume confirmation (1.5x) validates breakout strength without being too restrictive
# Session filter (08-20 UTC) reduces noise trades

name = "1h_4hDonchian20_1dADX25_VolumeConfirm"
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
    
    # Get 4h data ONCE before loop for Donchian calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:  # Need at least 20 completed 4h bars for Donchian(20)
        return np.zeros(n)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate 4h Donchian channels (20-period)
    # Upper = max(high, 20), Lower = min(low, 20)
    high_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_upper_4h = high_20
    donchian_lower_4h = low_20
    donchian_mid_4h = (donchian_upper_4h + donchian_lower_4h) / 2.0
    
    # Align 4h Donchian to 1h timeframe (wait for completed 4h bar)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper_4h)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower_4h)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_4h, donchian_mid_4h)
    
    # Get 1d data ONCE before loop for ADX25 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need at least 30 completed daily bars for ADX
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX(14)
    # ADX requires +DI, -DI, and DX
    # +DI = 100 * EWM of (+DM / TR), -DI = 100 * EWM of (-DM / TR)
    # DX = 100 * abs(+DI - -DI) / (+DI + -DI)
    # ADX = EWM of DX
    high_diff = np.diff(high_1d, prepend=high_1d[0])
    low_diff = np.diff(low_1d, prepend=low_1d[0])
    
    plus_dm = np.where((high_diff > -low_diff) & (high_diff > 0), high_diff, 0.0)
    minus_dm = np.where((-low_diff > high_diff) & (-low_diff > 0), -low_diff, 0.0)
    
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = np.abs(high_1d[0] - low_1d[0])  # Fix first value
    tr2[0] = np.abs(high_1d[0] - close_1d[0])  # Fix first value
    tr3[0] = np.abs(low_1d[0] - close_1d[0])  # Fix first value
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Avoid division by zero
    tr_mean = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / tr_mean
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / tr_mean
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_1d = adx
    
    # Align 1d ADX to 1h timeframe (wait for completed 1d bar)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 1h
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
            np.isnan(adx_1d_aligned[i]) or np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 4h Donchian upper, 1d ADX > 25 (trending), volume confirmation, in session
            if (close[i] > donchian_upper_aligned[i] and 
                adx_1d_aligned[i] > 25.0 and 
                volume_confirm[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below 4h Donchian lower, 1d ADX > 25 (trending), volume confirmation, in session
            elif (close[i] < donchian_lower_aligned[i] and 
                  adx_1d_aligned[i] > 25.0 and 
                  volume_confirm[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price crosses back below 4h Donchian midpoint
            if close[i] < donchian_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price crosses back above 4h Donchian midpoint
            if close[i] > donchian_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals