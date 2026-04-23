#!/usr/bin/env python3
"""
Hypothesis: 6h timeframe strategy using weekly Donchian channel (20) breakout with 1d ADX trend filter and volume confirmation.
- Uses 1w Donchian breakout for signal direction (avoids noise on lower timeframes)
- Uses 1d ADX > 25 to filter for trending markets only (works in bull/bear via trend strength)
- Volume confirmation (> 2.0x average) reduces false breakouts
- Position size: 0.25 (discrete level to minimize fee churn)
- Target: 12-37 trades/year (50-150 over 4 years) to avoid fee drag
- Weekly Donchian provides structural breaks that work in both bull and bear markets when combined with ADX trend filter
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
    
    # Weekly Donchian channel (20) for breakout signals
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 20-period Donchian channels
    high_20w = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    low_20w = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align to 6h timeframe (use prior completed weekly bar)
    high_20w_aligned = align_htf_to_ltf(prices, df_1w, high_20w)
    low_20w_aligned = align_htf_to_ltf(prices, df_1w, low_20w)
    
    # 1d ADX for trend filter (ADX > 25 = trending market)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = 0  # First period has no prior close
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = dm_minus[0] = 0  # First period
    
    # Smooth TR, DM+, DM- with Wilder's smoothing (equivalent to EMA with alpha=1/period)
    atr_1d = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_1d = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_1d = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Calculate DI+ and DI-
    di_plus_1d = 100 * dm_plus_1d / atr_1d
    di_minus_1d = 100 * dm_minus_1d / atr_1d
    
    # Calculate DX and ADX
    dx_1d = 100 * np.abs(di_plus_1d - di_minus_1d) / (di_plus_1d + di_minus_1d)
    adx_1d = pd.Series(dx_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align ADX to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume confirmation: > 2.0x 24-period average (strict for 6h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14, 24)  # Donchian20, ADX14, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(high_20w_aligned[i]) or
            np.isnan(low_20w_aligned[i]) or
            np.isnan(adx_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        # Donchian breakout signals (using current close vs prior weekly levels)
        breakout_up = close[i] > high_20w_aligned[i-1]  # Close above prior 20w high
        breakout_down = close[i] < low_20w_aligned[i-1]  # Close below prior 20w low
        
        # ADX trend filter (> 25 = trending market)
        trending = adx_1d_aligned[i] > 25
        
        if position == 0:
            # Long: 20w Donchian breakout up AND trending market AND volume confirmation
            if breakout_up and trending and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: 20w Donchian breakout down AND trending market AND volume confirmation
            elif breakout_down and trending and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: 20w Donchian break down OR ADX < 20 (trend weakening)
            if close[i] < low_20w_aligned[i] or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: 20w Donchian break up OR ADX < 20 (trend weakening)
            if close[i] > high_20w_aligned[i] or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyDonchian20_Breakout_1dADX25_VolumeSpike"
timeframe = "6h"
leverage = 1.0