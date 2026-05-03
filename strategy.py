#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian channel breakout with volume confirmation and 1d ADX regime filter.
# Long when price breaks above 4h Donchian upper channel (20-period) with volume > 1.5x 20-period average and 1d ADX > 25.
# Short when price breaks below 4h Donchian lower channel with volume confirmation and 1d ADX > 25.
# Uses 1h timeframe only for precise entry timing, signal direction from 4h structure.
# Session filter (08-20 UTC) to reduce noise. Position size: 0.20.
# Target: 15-37 trades/year (60-150 over 4 years) to minimize fee drag.

name = "1h_DonchianBreakout_4hVol_1dADX_Regime"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for Donchian channel
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Rolling max/min for Donchian channels
    high_roll = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align 4h Donchian channels to 1h timeframe
    donchian_high = align_htf_to_ltf(prices, df_4h, high_roll)
    donchian_low = align_htf_to_ltf(prices, df_4h, low_roll)
    
    # Get 1d data for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX (14-period) for regime filtering
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Plus Directional Movement (+DM) and Minus Directional Movement (-DM)
    dm_plus = np.concatenate([[np.nan], np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                                                 np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)])
    dm_minus = np.concatenate([[np.nan], np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                                                  np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)])
    
    # Smoothed TR, +DM, -DM (Wilder's smoothing)
    atr_1d = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_smoothed = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_smoothed = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Plus and Minus Directional Indicators
    plus_di_1d = 100 * dm_plus_smoothed / atr_1d
    minus_di_1d = 100 * dm_minus_smoothed / atr_1d
    
    # Directional Index and ADX
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    dx_1d = np.where((plus_di_1d + minus_di_1d) == 0, 0, dx_1d)
    adx_1d = pd.Series(dx_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align 1d ADX to 1h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume confirmation: 1.5x 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(adx_1d_aligned[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Regime filter: only trade when ADX > 25 (trending market)
        is_trending = adx_1d_aligned[i] > 25
        
        if position == 0:
            # Long: price breaks above Donchian high with volume confirmation and trending regime
            if close[i] > donchian_high[i] and volume_ok[i] and is_trending:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below Donchian low with volume confirmation and trending regime
            elif close[i] < donchian_low[i] and volume_ok[i] and is_trending:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price breaks below Donchian low OR reverse signal
            if close[i] < donchian_low[i] or (close[i] < donchian_low[i] and volume_ok[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price breaks above Donchian high OR reverse signal
            if close[i] > donchian_high[i] or (close[i] > donchian_high[i] and volume_ok[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals