#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R Extreme with 1d ADX Trend Filter and Volume Spike
- Primary timeframe: 6h (proven to work with Williams %R extremes)
- Williams %R(14) < -80 for oversold/long, > -20 for overbought/short
- 1d ADX(14) > 25 confirms strong trend (avoid ranging markets)
- Volume spike > 2.0 * 20-period MA confirms momentum
- Long: Williams %R < -80 AND ADX > 25 AND volume spike
- Short: Williams %R > -20 AND ADX > 25 AND volume spike
- Exit when Williams %R reverts to -50 (mean reversion)
- Discrete signal size: 0.25
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
- Works in bull/bear: ADX filter ensures we only trade strong trends, Williams %R catches pullbacks/extensions
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
    
    # Calculate 6h Williams %R(14)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 14:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_6h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_6h).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_6h) / (highest_high - lowest_low) * -100
    # Avoid division by zero
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    williams_r_aligned = align_htf_to_ltf(prices, df_6h, williams_r)
    
    # Calculate 1d ADX(14) for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = pd.Series(high_1d).shift(1) - pd.Series(low_1d)
    tr2 = pd.Series(high_1d) - pd.Series(close_1d).shift(1)
    tr3 = pd.Series(close_1d).shift(1) - pd.Series(low_1d)
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Movement
    up_move = pd.Series(high_1d).diff()
    down_move = -pd.Series(low_1d).diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr
    minus_di = 100 * minus_dm_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    # Avoid division by zero
    adx = np.where((plus_di + minus_di) == 0, 0, adx)
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(14, 20)  # Need Williams %R(14), ADX(14), volume MA(20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R < -80 (oversold) AND ADX > 25 (strong trend) AND volume spike
            if williams_r_aligned[i] < -80 and adx_aligned[i] > 25 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought) AND ADX > 25 (strong trend) AND volume spike
            elif williams_r_aligned[i] > -20 and adx_aligned[i] > 25 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R reverts to -50 (mean reversion)
            if williams_r_aligned[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R reverts to -50 (mean reversion)
            if williams_r_aligned[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_Extreme_1dADXTrend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0