#!/usr/bin/env python3
# 6h_1w_turtle_soup_v1
# Strategy: 6h Turtle Soup (false breakout fade) with 1w trend filter and volume confirmation
# Timeframe: 6h
# Leverage: 1.0
# Hypothesis: Turtle Soup fades false breakouts of the 20-period high/low. In trending markets (identified by 1w ADX > 25), these reversals offer high-probability entries. Volume confirmation ensures institutional participation. Works in both bull and bear markets by fading exhaustion moves.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_turtle_soup_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # 1w ADX(14) for trend filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    dm_plus = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w), 
                       np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)), 
                        np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr_period = 14
    atr = np.zeros_like(tr)
    dm_plus_smooth = np.zeros_like(dm_plus)
    dm_minus_smooth = np.zeros_like(dm_minus)
    
    # Initial smoothed values
    atr[tr_period-1] = np.mean(tr[:tr_period])
    dm_plus_smooth[tr_period-1] = np.mean(dm_plus[:tr_period])
    dm_minus_smooth[tr_period-1] = np.mean(dm_minus[:tr_period])
    
    # Wilder's smoothing
    for i in range(tr_period, len(tr)):
        atr[i] = (atr[i-1] * (tr_period - 1) + tr[i]) / tr_period
        dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (tr_period - 1) + dm_plus[i]) / tr_period
        dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (tr_period - 1) + dm_minus[i]) / tr_period
    
    # DI and DX
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    dx = np.zeros_like(di_plus)
    dx[tr_period:] = 100 * np.abs(di_plus[tr_period:] - di_minus[tr_period:]) / (di_plus[tr_period:] + di_minus[tr_period:])
    
    # ADX
    adx = np.zeros_like(dx)
    adx[2*tr_period-1:] = np.nan
    for i in range(2*tr_period-1, len(dx)):
        if i == 2*tr_period-1:
            adx[i] = np.mean(dx[tr_period:i+1])
        else:
            adx[i] = (adx[i-1] * (tr_period - 1) + dx[i]) / tr_period
    
    adx_1w = adx
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # 6h Donchian channels (20-period)
    donchian_len = 20
    high_max = pd.Series(high).rolling(window=donchian_len, min_periods=donchian_len).max().values
    low_min = pd.Series(low).rolling(window=donchian_len, min_periods=donchian_len).min().values
    
    # Volume confirmation: volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_ratio = pd.Series(volume) / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(donchian_len, n):
        # Skip if any required data is invalid
        if (np.isnan(adx_1w_aligned[i]) or np.isnan(high_max[i]) or 
            np.isnan(low_min[i]) or np.isnan(vol_ratio.iloc[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Only trade in trending markets (ADX > 25)
        trending = adx_1w_aligned[i] > 25
        
        # Volume confirmation
        vol_confirmed = vol_ratio.iloc[i] > 1.3
        
        # Turtle Soup logic: fade false breakouts
        # Long setup: false breakdown below 20-period low
        if trending and vol_confirmed and low[i] < low_min[i] and close[i] > low_min[i] and position != 1:
            # Price broke below Donchian low but closed back above - long signal
            position = 1
            signals[i] = 0.25
        # Short setup: false breakout above 20-period high
        elif trending and vol_confirmed and high[i] > high_max[i] and close[i] < high_max[i] and position != -1:
            # Price broke above Donchian high but closed back below - short signal
            position = -1
            signals[i] = -0.25
        # Exit: price reaches opposite Donchian band or trend weakens
        elif position == 1 and (close[i] >= high_max[i] or adx_1w_aligned[i] < 20):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] <= low_min[i] or adx_1w_aligned[i] < 20):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals