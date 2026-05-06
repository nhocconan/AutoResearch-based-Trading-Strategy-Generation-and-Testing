#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h Donchian breakout with volume confirmation and 1d ADX trend filter
# - Long when price breaks above 12h Donchian(20) high with volume expansion and 1d ADX > 25
# - Short when price breaks below 12h Donchian(20) low with volume expansion and 1d ADX > 25
# - Exit when price crosses 12h Donchian midline (mean of high/low over 20 periods)
# - Volume expansion defined as current volume > 1.5x 20-period average
# - Designed to capture strong trends while avoiding choppy markets via ADX filter
# - Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position sizing

name = "4h_DonchianBreakout_12hVolume_ADXTrend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Donchian channels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h Donchian channels (20-period high/low)
    high_12h = df_12h['high'].rolling(window=20, min_periods=20).max().values
    low_12h = df_12h['low'].rolling(window=20, min_periods=20).min().values
    mid_12h = (high_12h + low_12h) / 2.0
    
    # Align 12h Donchian levels to 4h timeframe
    high_12h_4h = align_htf_to_ltf(prices, df_12h, high_12h)
    low_12h_4h = align_htf_to_ltf(prices, df_12h, low_12h)
    mid_12h_4h = align_htf_to_ltf(prices, df_12h, mid_12h)
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1d ADX (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    
    # Smooth TR, DM+, DM- (14-period)
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    # DI+ and DI-
    di_plus = np.where(tr14 > 0, 100 * dm_plus_14 / tr14, 0)
    di_minus = np.where(tr14 > 0, 100 * dm_minus_14 / tr14, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) > 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align 1d ADX to 4h timeframe
    adx_4h = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume filters
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_expansion = volume > (1.5 * vol_ma_20)  # Volume expansion
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if any critical value is NaN
        if (np.isnan(high_12h_4h[i]) or np.isnan(low_12h_4h[i]) or np.isnan(mid_12h_4h[i]) or
            np.isnan(adx_4h[i]) or np.isnan(volume_expansion[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price breaks above 12h Donchian high with volume expansion and ADX > 25
            if close[i] > high_12h_4h[i] and volume_expansion[i] and adx_4h[i] > 25:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below 12h Donchian low with volume expansion and ADX > 25
            elif close[i] < low_12h_4h[i] and volume_expansion[i] and adx_4h[i] > 25:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below 12h Donchian midline
            if close[i] < mid_12h_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above 12h Donchian midline
            if close[i] > mid_12h_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals