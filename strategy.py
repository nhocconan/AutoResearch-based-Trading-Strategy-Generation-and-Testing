#!/usr/bin/env python3
"""
12h strategy using 1-day Donchian channel breakout with volume confirmation and 1-week ADX trend filter.
Long when price breaks above 20-period high with volume surge and ADX > 25.
Short when price breaks below 20-period low with volume surge and ADX > 25.
Exit when price returns to 20-period midpoint or ADX < 20.
Designed for low turnover: ~15-25 trades/year per symbol to minimize fee drift.
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
    
    # Load 1-day data once for Donchian channel
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Donchian channel (20)
    donch_period = 20
    upper = pd.Series(high_1d).rolling(window=donch_period, min_periods=donch_period).max().values
    lower = pd.Series(low_1d).rolling(window=donch_period, min_periods=donch_period).min().values
    midpoint = (upper + lower) / 2
    
    # Load 1-week data for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ADX (14)
    adx_period = 14
    
    # True Range
    tr1_w = high_1w - low_1w
    tr2_w = np.abs(high_1w - np.roll(close_1w, 1))
    tr3_w = np.abs(low_1w - np.roll(close_1w, 1))
    tr_w = np.maximum(tr1_w, np.maximum(tr2_w, tr3_w))
    tr_w[0] = tr1_w[0]
    atr_w = pd.Series(tr_w).ewm(span=adx_period, adjust=False, min_periods=adx_period).mean().values
    
    # Directional Movement
    up_move = high_1w - np.roll(high_1w, 1)
    down_move = np.roll(low_1w, 1) - low_1w
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=adx_period, adjust=False, min_periods=adx_period).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=adx_period, adjust=False, min_periods=adx_period).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr_w
    minus_di = 100 * minus_dm_smooth / atr_w
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) > 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = pd.Series(dx).ewm(span=adx_period, adjust=False, min_periods=adx_period).mean().values
    
    # Volume filter: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(50, n):
        # 1-day index (2 bars per day for 12h timeframe)
        idx_1d = i // 2
        if idx_1d < donch_period:
            continue
        
        # Use previous 1d values to avoid look-ahead
        prev_idx = idx_1d - 1
        if prev_idx < 0:
            continue
            
        # Get Donchian values from previous 1d bar
        upper_prev = upper[prev_idx] if prev_idx < len(upper) else upper[-1]
        lower_prev = lower[prev_idx] if prev_idx < len(lower) else lower[-1]
        midpoint_prev = midpoint[prev_idx] if prev_idx < len(midpoint) else midpoint[-1]
        
        # 1-week index (14 bars per week for 12h timeframe: 7*2=14)
        idx_1w = i // 14
        if idx_1w < adx_period:
            continue
        
        # Use previous 1w values to avoid look-ahead
        prev_idx_1w = idx_1w - 1
        if prev_idx_1w < 0:
            continue
            
        # Get ADX from previous 1w bar
        adx_prev = adx[prev_idx_1w] if prev_idx_1w < len(adx) else adx[-1]
        
        if np.isnan(upper_prev) or np.isnan(lower_prev) or np.isnan(midpoint_prev) or np.isnan(adx_prev):
            continue
        
        # Create arrays for alignment
        upper_arr = np.full(len(df_1d), upper_prev)
        lower_arr = np.full(len(df_1d), lower_prev)
        midpoint_arr = np.full(len(df_1d), midpoint_prev)
        adx_arr = np.full(len(df_1w), adx_prev)
        
        upper_12h = align_htf_to_ltf(prices, df_1d, upper_arr)[i]
        lower_12h = align_htf_to_ltf(prices, df_1d, lower_arr)[i]
        midpoint_12h = align_htf_to_ltf(prices, df_1d, midpoint_arr)[i]
        adx_12h = align_htf_to_ltf(prices, df_1w, adx_arr)[i]
        
        if np.isnan(upper_12h) or np.isnan(lower_12h) or np.isnan(midpoint_12h) or np.isnan(adx_12h):
            continue
        
        if position == 0:
            # Long: Price breaks above upper Donchian with volume surge and ADX > 25
            if close[i] > upper_12h and volume[i] > vol_ma[i] * 1.5 and adx_12h > 25:
                position = 1
                signals[i] = position_size
            # Short: Price breaks below lower Donchian with volume surge and ADX > 25
            elif close[i] < lower_12h and volume[i] > vol_ma[i] * 1.5 and adx_12h > 25:
                position = -1
                signals[i] = -position_size
        elif position == 1:
            # Exit: Price returns to midpoint or ADX < 20
            if close[i] <= midpoint_12h or adx_12h < 20:
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Exit: Price returns to midpoint or ADX < 20
            if close[i] >= midpoint_12h or adx_12h < 20:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "12h_1d_Donchian_1wADX"
timeframe = "12h"
leverage = 1.0