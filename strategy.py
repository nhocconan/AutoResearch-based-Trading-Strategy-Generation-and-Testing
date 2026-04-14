#!/usr/bin/env python3
"""
Hypothesis: 4-hour strategy using 1-day Keltner Channel breakout with volume confirmation and 1-week ADX trend filter.
Long when price closes above upper KC (20,2) with volume > 1.5x average and 1-week ADX > 25.
Short when price closes below lower KC (20,2) with volume > 1.5x average and 1-week ADX > 25.
Exit when price returns to middle KC or 1-week ADX < 20.
Designed for low turnover: ~20-30 trades/year per symbol to minimize fee drag.
Uses proven Keltner breakout with ADX trend filter to work in trending markets while avoiding chop.
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
    
    # Load 1-day data once for Keltner Channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1-day Keltner Channels (20,2)
    kc_period = 20
    kc_mult = 2
    
    # Typical Price
    tp = (high_1d + low_1d + close_1d) / 3
    
    # EMA of Typical Price (middle line)
    ema_tp = pd.Series(tp).ewm(span=kc_period, adjust=False, min_periods=kc_period).mean().values
    
    # ATR
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    atr = pd.Series(tr).ewm(span=kc_period, adjust=False, min_periods=kc_period).mean().values
    
    upper = ema_tp + kc_mult * atr
    lower = ema_tp - kc_mult * atr
    middle = ema_tp
    
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
        # 1-day index (6 bars per day for 4h timeframe? Wait: 24h/4h = 6 bars per day)
        # Actually: 4h timeframe means 6 bars per 24h day
        idx_1d = i // 6
        if idx_1d < kc_period:
            continue
        
        # Use previous 1d values to avoid look-ahead (previous completed bar)
        prev_idx = idx_1d - 1
        if prev_idx < 0:
            continue
            
        # Get Keltner Channels from previous 1d bar
        upper_prev = upper[prev_idx] if prev_idx < len(upper) else upper[-1]
        lower_prev = lower[prev_idx] if prev_idx < len(lower) else lower[-1]
        middle_prev = middle[prev_idx] if prev_idx < len(middle) else middle[-1]
        
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
        
        if np.isnan(upper_prev) or np.isnan(lower_prev) or np.isnan(middle_prev) or np.isnan(adx_prev):
            continue
        
        # Create arrays for alignment (using previous values)
        upper_arr = np.full(len(df_1d), upper_prev)
        lower_arr = np.full(len(df_1d), lower_prev)
        middle_arr = np.full(len(df_1d), middle_prev)
        adx_arr = np.full(len(df_1w), adx_prev)
        
        upper_4h = align_htf_to_ltf(prices, df_1d, upper_arr)[i]
        lower_4h = align_htf_to_ltf(prices, df_1d, lower_arr)[i]
        middle_4h = align_htf_to_ltf(prices, df_1d, middle_arr)[i]
        adx_4h = align_htf_to_ltf(prices, df_1w, adx_arr)[i]
        
        if position == 0:
            # Long: price closes above upper KC + volume surge + 1w trending (ADX > 25)
            if (close[i] > upper_4h and 
                volume[i] > vol_ma[i] * 1.5 and
                adx_4h > 25):
                position = 1
                signals[i] = position_size
            # Short: price closes below lower KC + volume surge + 1w trending (ADX > 25)
            elif (close[i] < lower_4h and 
                  volume[i] > vol_ma[i] * 1.5 and
                  adx_4h > 25):
                position = -1
                signals[i] = -position_size
        elif position == 1:
            # Exit: price returns to middle KC or 1w trend weakens (ADX < 20)
            if close[i] < middle_4h or adx_4h < 20:
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Exit: price returns to middle KC or 1w trend weakens (ADX < 20)
            if close[i] > middle_4h or adx_4h < 20:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "4h_1d_Keltner_Breakout_1wADX"
timeframe = "4h"
leverage = 1.0