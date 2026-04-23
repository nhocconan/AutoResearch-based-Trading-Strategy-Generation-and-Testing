#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Index (Bull Power/Bear Power) with 1d ADX regime filter and volume confirmation.
Long when Bull Power > 0 AND Bear Power < 0 (bullish momentum) AND 1d ADX > 25 (trending) AND volume > 1.5x 20-period MA.
Short when Bear Power < 0 AND Bull Power < 0 (bearish momentum) AND 1d ADX > 25 (trending) AND volume > 1.5x 20-period MA.
Exit when Elder Ray signals weaken (Bull Power <= 0 for long, Bear Power >= 0 for short) OR 1d ADX < 20 (range regime) OR volume filter fails.
Elder Ray measures bull/bear power via EMA13; ADX filters for trending markets only to avoid whipsaws; volume confirms conviction.
Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
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
    
    # Calculate 6h EMA13 for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate 6h Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Calculate 1d ADX for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX(14)
    # True Range
    tr1 = pd.Series(high_1d).shift(1) - pd.Series(low_1d).shift(1)
    tr2 = abs(pd.Series(high_1d).shift(1) - pd.Series(close_1d).shift(1))
    tr3 = abs(pd.Series(low_1d).shift(1) - pd.Series(close_1d).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).values
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Movement
    up_move = pd.Series(high_1d).diff().values
    down_move = -pd.Series(low_1d).diff().values
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr_1d
    minus_di = 100 * minus_dm_smooth / atr_1d
    
    # DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx_1d = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 6h volume MA (20-period) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(13, 30, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: 6h volume > 1.5x 20-period MA
        vol_filter = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: Bull Power > 0 AND Bear Power < 0 (bullish) AND ADX > 25 (trending) AND volume filter
            if bull_power[i] > 0 and bear_power[i] < 0 and adx_1d_aligned[i] > 25 and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 AND Bull Power < 0 (bearish) AND ADX > 25 (trending) AND volume filter
            elif bear_power[i] < 0 and bull_power[i] < 0 and adx_1d_aligned[i] > 25 and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: Bull Power <= 0 (weakening bullish) OR ADX < 20 (range) OR volume filter fails
                if bull_power[i] <= 0 or adx_1d_aligned[i] < 20 or not vol_filter:
                    exit_signal = True
            elif position == -1:
                # Short exit: Bear Power >= 0 (weakening bearish) OR ADX < 20 (range) OR volume filter fails
                if bear_power[i] >= 0 or adx_1d_aligned[i] < 20 or not vol_filter:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_ElderRay_ADXRegime_VolumeFilter"
timeframe = "6h"
leverage = 1.0