#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with 1-day ADX(14) trend filter and volume confirmation.
# Uses Donchian channel breakouts as entry signals, filtered by 1-day ADX > 25 for trending markets
# and volume > 1.5x 20-period average for confirmation. Exits when price crosses opposite Donchian band
# or ADX drops below 20. Works in both bull (breakouts up) and bear (breakouts down) markets.
# Target: 20-50 trades/year with position size 0.25.

name = "4h_Donchian20_ADX14_Volume_Trend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1-day ADX(14) for trend strength
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # True Range components
    prev_close = np.roll(df_1d['close'], 1)
    prev_close[0] = np.nan
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - prev_close)
    tr3 = np.abs(df_1d['low'] - prev_close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    high_diff = df_1d['high'].diff()
    low_diff = -df_1d['low'].diff()  # negative of low diff
    
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0.0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0.0)
    
    # Smoothed values
    tr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_dm_14 = pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    minus_dm_14 = pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_14 / tr_14
    minus_di = 100 * minus_dm_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # ADX trend filters
    adx_trending = adx > 25
    adx_not_trending = adx < 20
    
    # Align ADX indicators to 4h timeframe
    adx_trending_aligned = align_htf_to_ltf(prices, df_1d, adx_trending)
    adx_not_trending_aligned = align_htf_to_ltf(prices, df_1d, adx_not_trending)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need enough data for Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(adx_trending_aligned[i]) or np.isnan(adx_not_trending_aligned[i]) or
            np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Donchian breakout up + ADX trending + volume confirmation
            if (close[i] > donchian_high[i-1]) and adx_trending_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: Donchian breakout down + ADX trending + volume confirmation
            elif (close[i] < donchian_low[i-1]) and adx_trending_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Donchian breakout down OR ADX loses trend
            if (close[i] < donchian_low[i-1]) or adx_not_trending_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Donchian breakout up OR ADX loses trend
            if (close[i] > donchian_high[i-1]) or adx_not_trending_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals