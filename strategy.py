#!/usr/bin/env python3
"""
Hypothesis: 4-hour strategy using 1-day Williams %R reversal signals with 1-week ADX trend filter.
Long when daily Williams %R crosses above -80 (oversold) with ADX > 25.
Short when daily Williams %R crosses below -20 (overbought) with ADX > 25.
Exit when Williams %R returns to -50 (neutral) or ADX < 20.
Williams %R identifies reversal points in extended moves; ADX filters for trending conditions.
Designed for low turnover: ~20-30 trades/year per symbol to minimize fee drag.
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
    
    # Load 1-day data once for Williams %R
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams %R (14)
    wr_period = 14
    
    # Highest high and lowest low over period
    highest_high = pd.Series(high_1d).rolling(window=wr_period, min_periods=wr_period).max().values
    lowest_low = pd.Series(low_1d).rolling(window=wr_period, min_periods=wr_period).min().values
    
    # Williams %R: -100 * (HH - Close) / (HH - LL)
    wr = np.where((highest_high - lowest_low) != 0,
                  -100 * (highest_high - close_1d) / (highest_high - lowest_low),
                  -50)  # Neutral when range is zero
    
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
        # 1-day index (6 bars per day for 4h timeframe)
        idx_1d = i // 6
        if idx_1d < wr_period:
            continue
        
        # Use previous 1d values to avoid look-ahead
        prev_idx = idx_1d - 1
        if prev_idx < 0:
            continue
            
        # Get Williams %R from previous 1d bar
        wr_prev = wr[prev_idx] if prev_idx < len(wr) else wr[-1]
        wr_prev2 = wr[prev_idx-1] if prev_idx-1 >= 0 else wr[0]  # For crossover detection
        
        # 1-week index (42 bars per week for 4h timeframe: 7*6=42)
        idx_1w = i // 42
        if idx_1w < adx_period:
            continue
        
        # Use previous 1w values to avoid look-ahead
        prev_idx_1w = idx_1w - 1
        if prev_idx_1w < 0:
            continue
            
        # Get ADX from previous 1w bar
        adx_prev = adx[prev_idx_1w] if prev_idx_1w < len(adx) else adx[-1]
        
        if np.isnan(wr_prev) or np.isnan(wr_prev2) or np.isnan(adx_prev):
            continue
        
        # Create arrays for alignment
        wr_arr = np.full(len(df_1d), wr_prev)
        wr_prev2_arr = np.full(len(df_1d), wr_prev2)
        adx_arr = np.full(len(df_1w), adx_prev)
        
        wr_4h = align_htf_to_ltf(prices, df_1d, wr_arr)[i]
        wr_prev2_4h = align_htf_to_ltf(prices, df_1d, wr_prev2_arr)[i]
        adx_4h = align_htf_to_ltf(prices, df_1w, adx_arr)[i]
        
        if np.isnan(wr_4h) or np.isnan(wr_prev2_4h) or np.isnan(adx_4h):
            continue
        
        if position == 0:
            # Long: Williams %R crosses above -80 (oversold reversal) + volume surge + ADX > 25
            if (wr_prev2_4h <= -80 and wr_4h > -80 and  # Crossover above -80
                volume[i] > vol_ma[i] * 1.5 and
                adx_4h > 25):
                position = 1
                signals[i] = position_size
            # Short: Williams %R crosses below -20 (overbought reversal) + volume surge + ADX > 25
            elif (wr_prev2_4h >= -20 and wr_4h < -20 and  # Crossover below -20
                  volume[i] > vol_ma[i] * 1.5 and
                  adx_4h > 25):
                position = -1
                signals[i] = -position_size
        elif position == 1:
            # Exit: Williams %R returns to -50 (neutral) or ADX < 20
            if wr_4h >= -50 or adx_4h < 20:
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Exit: Williams %R returns to -50 (neutral) or ADX < 20
            if wr_4h <= -50 or adx_4h < 20:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "4h_1d_WilliamsR_1wADX"
timeframe = "4h"
leverage = 1.0