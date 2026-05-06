#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1-day Donchian channel breakouts with volume confirmation and ADX trend filter
# - Long when price breaks above 1-day Donchian high (20-period) with volume expansion and ADX > 25
# - Short when price breaks below 1-day Donchian low (20-period) with volume expansion and ADX > 25
# - Exit when price returns to the 1-day Donchian midpoint or reverses with volume
# - Uses 1-day ADX for trend strength filtering to avoid whipsaws in ranging markets
# - Position sizing: 0.25 for strong signals, 0.125 for weaker signals
# - Designed to capture major trends while avoiding false breakouts in low volatility environments
# - Target: 50-150 total trades over 4 years (12-37/year) with controlled frequency

name = "12h_Donchian20_1dADX_Volume_Breakout"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-day data for Donchian channels and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1-day Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Donchian high and low
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Calculate 1-day ADX for trend strength
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # First value has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    # DI+ and DI-
    plus_di = 100 * dm_plus_14 / tr14
    minus_di = 100 * dm_minus_14 / tr14
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = np.where((plus_di + minus_di) == 0, 0, dx)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align 1-day indicators to 12h timeframe
    donchian_high_12h = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_12h = align_htf_to_ltf(prices, df_1d, donchian_low)
    donchian_mid_12h = align_htf_to_ltf(prices, df_1d, donchian_mid)
    adx_12h = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume filters
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)  # Volume confirmation
    volume_expansion = volume > np.roll(volume, 1)  # Current volume > previous
    volume_expansion[0] = False
    
    # Session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after warmup
        # Skip if any critical value is NaN or outside session
        if (np.isnan(donchian_high_12h[i]) or np.isnan(donchian_low_12h[i]) or 
            np.isnan(donchian_mid_12h[i]) or np.isnan(adx_12h[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(volume_expansion[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above Donchian high with volume and trend
            if (close[i] > donchian_high_12h[i] and 
                volume_expansion[i] and 
                adx_12h[i] > 25):
                signals[i] = 0.25
                position = 1
            # Short breakout: price breaks below Donchian low with volume and trend
            elif (close[i] < donchian_low_12h[i] and 
                  volume_expansion[i] and 
                  adx_12h[i] > 25):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to midpoint or reverses with volume
            if close[i] <= donchian_mid_12h[i]:
                signals[i] = 0.0
                position = 0
            elif (close[i] < donchian_low_12h[i] and 
                  volume_expansion[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to midpoint or reverses with volume
            if close[i] >= donchian_mid_12h[i]:
                signals[i] = 0.0
                position = 0
            elif (close[i] > donchian_high_12h[i] and 
                  volume_expansion[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals