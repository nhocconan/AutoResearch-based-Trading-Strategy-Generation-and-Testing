#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_1dADXTrend_VolumeSpike_HTF
Hypothesis: Donchian(20) breakout on 6h filtered by 1d ADX>25 (strong trend) and volume spike (>2x 20-bar average).
Only breakouts in the direction of the 1d trend are taken. ADX ensures we trade strong trending markets,
while volume spike confirms institutional participation. Works in both bull and bear markets by only
trading in the direction of the 1d trend. Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d ADX for trend strength filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # First bar
    tr2[0] = np.abs(high_1d[0] - close_1d[0])  # First bar
    tr3[0] = np.abs(low_1d[0] - close_1d[0])   # First bar
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed TR, +DM, -DM (Wilder's smoothing = EMA with alpha=1/period)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align 1d ADX to 6h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 6h Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need enough for Donchian (20), volume avg (20), and ADX (14+14+14)
    start_idx = max(100, 20, 20, 30)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        adx_val = adx_aligned[i]
        vol_spike = volume_spike[i]
        size = 0.25  # 25% position size
        
        if position == 0:
            # Flat - look for entry: Donchian breakout in direction of 1d trend with volume spike and ADX>25
            # Long: price breaks above Donchian high AND ADX>25 (strong trend) AND volume spike
            # Short: price breaks below Donchian low AND ADX>25 (strong trend) AND volume spike
            long_breakout = close_val > donchian_high[i]
            short_breakout = close_val < donchian_low[i]
            strong_trend = adx_val > 25
            
            if long_breakout and strong_trend and vol_spike:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_breakout and strong_trend and vol_spike:
                signals[i] = -size
                position = -1
                entry_price = close_val
        elif position == 1:
            # Long - exit when price breaks below Donchian low (failed breakout/trend reversal)
            if close_val < donchian_low[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when price breaks above Donchian high (failed breakout/trend reversal)
            if close_val > donchian_high[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Donchian20_Breakout_1dADXTrend_VolumeSpike_HTF"
timeframe = "6h"
leverage = 1.0