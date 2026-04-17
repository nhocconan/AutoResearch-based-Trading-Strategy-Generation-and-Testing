#!/usr/bin/env python3
"""
Hypothesis:
This strategy uses 6h Donchian breakouts (20-period) combined with 1-day volume spike (2x) and ADX trend filter (25).
In trending markets, breakouts with volume and trend confirmation tend to continue. Volume spike ensures conviction,
ADX > 25 filters for trending regimes, avoiding false breakouts in ranging markets.
Designed for 6h timeframe to achieve 12-37 trades/year with low decay. Works in both bull and bear markets by
filtering for trending regimes and using volume as confirmation of institutional interest.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_donchian(high, low, window):
    """Calculate Donchian channels"""
    upper = pd.Series(high).rolling(window=window, min_periods=window).max().values
    lower = pd.Series(low).rolling(window=window, min_periods=window).min().values
    return upper, lower

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h Donchian Channel (20-period) ===
    donch_hi, donch_lo = calculate_donchian(high, low, 20)
    
    # === 1-day Data ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # === 1-day ADX (14-period) for trend filter ===
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high_1d - np.concatenate([[high_1d[0]], high_1d[:-1]])
    down_move = np.concatenate([[low_1d[0]], low_1d[:-1]]) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    plus_dm_14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values
    minus_dm_14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_14 / (tr_14 + 1e-10)
    minus_di = 100 * minus_dm_14 / (tr_14 + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # === 1-day Volume Spike (vs 20-period average) ===
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 50
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(donch_hi[i]) or np.isnan(donch_lo[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i]) or
            np.isnan(volume_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Volume spike: current 1d volume > 2x 20-period average
        vol_spike = volume_1d_aligned[i] > vol_ma_20_1d_aligned[i] * 2.0
        
        # Trend filter: ADX > 25
        trend_filter = adx_aligned[i] > 25
        
        # Donchian breakout conditions
        breakout_up = close[i] > donch_hi[i-1]  # Break above upper band
        breakout_down = close[i] < donch_lo[i-1]  # Break below lower band
        
        # Entry logic: only enter when flat
        if position == 0:
            if vol_spike and trend_filter:
                # Long: breakout up
                if breakout_up:
                    signals[i] = 0.25
                    position = 1
                    continue
                # Short: breakout down
                elif breakout_down:
                    signals[i] = -0.25
                    position = -1
                    continue
        
        # Exit logic: exit when price returns to Donchian mid-point or conditions fail
        elif position == 1:
            donch_mid = (donch_hi[i] + donch_lo[i]) / 2
            # Exit long if price falls below midpoint or conditions fail
            if close[i] < donch_mid or not vol_spike or not trend_filter:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            donch_mid = (donch_hi[i] + donch_lo[i]) / 2
            # Exit short if price rises above midpoint or conditions fail
            if close[i] > donch_mid or not vol_spike or not trend_filter:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_1dVolume2x_ADX25"
timeframe = "6h"
leverage = 1.0