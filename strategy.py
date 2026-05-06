#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Donchian breakout with volume confirmation and ADX trend filter
# - Long when price breaks above 1d Donchian upper channel (20) with volume > 1.5x 20-period average
# - Short when price breaks below 1d Donchian lower channel (20) with volume > 1.5x 20-period average
# - Only take trades when 1d ADX > 25 (trending market)
# - Exit when price returns to 1d Donchian middle (mean of upper/lower) or reverses with volume
# - Designed to capture strong trends in both bull and bear markets with low trade frequency
# - Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position sizing

name = "12h_Donchian20_1dADX25_VolumeBreakout"
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
    
    # Get 1d data for Donchian channels and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
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
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smooth TR and DM
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    dm_plus14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    # DI+ and DI-
    di_plus = np.where(tr14 != 0, 100 * dm_plus14 / tr14, 0)
    di_minus = np.where(tr14 != 0, 100 * dm_minus14 / tr14, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align 1d indicators to 12h timeframe
    donchian_high_12h = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_12h = align_htf_to_ltf(prices, df_1d, donchian_low)
    donchian_mid_12h = align_htf_to_ltf(prices, df_1d, donchian_mid)
    adx_12h = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after warmup
        # Skip if any critical value is NaN or outside session
        if (np.isnan(donchian_high_12h[i]) or np.isnan(donchian_low_12h[i]) or
            np.isnan(donchian_mid_12h[i]) or np.isnan(adx_12h[i]) or
            np.isnan(volume_filter[i]) or not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: break above Donchian high with volume and ADX > 25
            if close[i] > donchian_high_12h[i] and volume_filter[i] and adx_12h[i] > 25:
                signals[i] = 0.25
                position = 1
            # Short entry: break below Donchian low with volume and ADX > 25
            elif close[i] < donchian_low_12h[i] and volume_filter[i] and adx_12h[i] > 25:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Donchian mid or breaks below Donchian low
            if close[i] <= donchian_mid_12h[i]:
                signals[i] = 0.0
                position = 0
            elif close[i] < donchian_low_12h[i]:  # Stop loss
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Donchian mid or breaks above Donchian high
            if close[i] >= donchian_mid_12h[i]:
                signals[i] = 0.0
                position = 0
            elif close[i] > donchian_high_12h[i]:  # Stop loss
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals