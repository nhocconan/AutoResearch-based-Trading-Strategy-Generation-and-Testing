#!/usr/bin/env python3
"""
Hypothesis: 6h timeframe with weekly Donchian(20) breakout + daily ADX(14) trend filter + 6h volume confirmation.
Long when price breaks above weekly Donchian high with daily ADX > 25 (trending) and 6h volume > 1.5x 20-period 6h volume average.
Short when price breaks below weekly Donchian low with daily ADX > 25 (trending) and volume confirmation.
Uses discrete position sizing 0.25 to limit fee drag. Target: 50-150 total trades over 4 years.
Weekly Donchian provides structural breakout levels; daily ADX filters for trending markets only; volume confirms participation.
Designed to work in bull markets (breakout continuation) and bear markets (strong trend continuation).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian channels
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Get daily data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly Donchian(20) channels
    def donchian_channel(high_vals, low_vals, window):
        upper = pd.Series(high_vals).rolling(window=window, min_periods=window).max().values
        lower = pd.Series(low_vals).rolling(window=window, min_periods=window).min().values
        return upper, lower
    
    donchian_upper_20w, donchian_lower_20w = donchian_channel(high_1w, low_1w, 20)
    
    # Calculate daily ADX(14) for trend strength
    def calculate_adx(high_vals, low_vals, close_vals, window):
        plus_dm = np.zeros_like(high_vals)
        minus_dm = np.zeros_like(high_vals)
        tr = np.zeros_like(high_vals)
        
        for i in range(1, len(high_vals)):
            high_diff = high_vals[i] - high_vals[i-1]
            low_diff = low_vals[i-1] - low_vals[i]
            
            plus_dm[i] = high_diff if high_diff > low_diff and high_diff > 0 else 0
            minus_dm[i] = low_diff if low_diff > high_diff and low_diff > 0 else 0
            tr[i] = max(high_vals[i] - low_vals[i], 
                       abs(high_vals[i] - close_vals[i-1]), 
                       abs(low_vals[i] - close_vals[i-1]))
        
        # Wilder's smoothing (alpha = 1/window)
        atr = pd.Series(tr).ewm(alpha=1/window, adjust=False, min_periods=window).mean().values
        plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/window, adjust=False, min_periods=window).mean().values / atr
        minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/window, adjust=False, min_periods=window).mean().values / atr
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = pd.Series(dx).ewm(alpha=1/window, adjust=False, min_periods=window).mean().values
        return adx
    
    adx_14_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Calculate 6h volume 20-period average
    vol_ma_20_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align all to primary timeframe (6h)
    donchian_upper_20w_aligned = align_htf_to_ltf(prices, df_1w, donchian_upper_20w)
    donchian_lower_20w_aligned = align_htf_to_ltf(prices, df_1w, donchian_lower_20w)
    adx_14_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_14_1d)
    vol_ma_20_6h_aligned = align_htf_to_ltf(prices, prices, vol_ma_20_6h)  # same timeframe
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # need enough for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_upper_20w_aligned[i]) or np.isnan(donchian_lower_20w_aligned[i]) or 
            np.isnan(adx_14_1d_aligned[i]) or np.isnan(vol_ma_20_6h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20_6h_aligned[i]
        # Trend filter: ADX > 25 indicates trending market
        trending = adx_14_1d_aligned[i] > 25
        
        if position == 0:
            # Long: price breaks above weekly Donchian high with trend and volume
            if (close[i] > donchian_upper_20w_aligned[i] and 
                trending and 
                volume_confirmed):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly Donchian low with trend and volume
            elif (close[i] < donchian_lower_20w_aligned[i] and 
                  trending and 
                  volume_confirmed):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below weekly Donchian low (breakdown)
            if close[i] < donchian_lower_20w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above weekly Donchian high (breakout)
            if close[i] > donchian_upper_20w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1wDonchian20_1dADX14_Volume_Confirm"
timeframe = "6h"
leverage = 1.0