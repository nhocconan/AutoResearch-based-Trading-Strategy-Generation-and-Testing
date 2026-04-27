#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dVolume_Trend_v2
Hypothesis: Donchian(20) breakout with 1d volume confirmation and trend filter provides robust signals across bull and bear markets. 
Uses 1d ADX > 25 for trend strength and requires volume > 1.5x 20-period average to confirm breakouts.
Target: 20-40 trades/year on 4h timeframe to minimize fee drag while capturing major trends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume and trend filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d volume average
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = vol_1d / np.where(vol_ma_1d == 0, 1, vol_ma_1d)
    
    # Calculate 1d ADX for trend strength
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth TR, +DM, -DM
    tr_period = 14
    atr = pd.Series(tr).ewm(alpha=1/tr_period, adjust=False).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/tr_period, adjust=False).mean().values / np.where(atr == 0, 1, atr)
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/tr_period, adjust=False).mean().values / np.where(atr == 0, 1, atr)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = np.where((plus_di + minus_di) == 0, 0, dx)
    adx = pd.Series(dx).ewm(alpha=1/tr_period, adjust=False).mean().values
    
    # Align 1d indicators to 4h timeframe
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Start after Donchian warmup
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(vol_ratio_aligned[i]) or np.isnan(adx_aligned[i]) or np.isnan(high_20[i]) or np.isnan(low_20[i]):
            signals[i] = 0.0
            continue
        
        vol_ratio = vol_ratio_aligned[i]
        adx_val = adx_aligned[i]
        
        if position == 0:
            # Long: price breaks above Donchian high, volume confirmation, strong trend
            if close[i] > high_20[i] and vol_ratio > 1.5 and adx_val > 25:
                signals[i] = size
                position = 1
            # Short: price breaks below Donchian low, volume confirmation, strong trend
            elif close[i] < low_20[i] and vol_ratio > 1.5 and adx_val > 25:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: price crosses below Donchian low or trend weakens
            if close[i] < low_20[i] or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above Donchian high or trend weakens
            if close[i] > high_20[i] or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian20_Breakout_1dVolume_Trend_v2"
timeframe = "4h"
leverage = 1.0