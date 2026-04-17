#!/usr/bin/env python3
"""
4h Donchian(20) breakout + volume confirmation + ADX filter
Long: break above upper band + volume > 1.5x 20-period volume SMA + ADX > 25
Short: break below lower band + volume > 1.5x 20-period volume SMA + ADX > 25
Exit: opposite signal or ADX < 20
Uses Donchian channel for breakout direction, volume for confirmation, ADX for trend strength.
Designed to work in both bull and bear markets by filtering with ADX > 25.
Target: 50-150 total trades over 4 years (12-37/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate ADX (14-period)
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value has no previous close
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
    atr = np.zeros_like(tr)
    plus_dm_smooth = np.zeros_like(plus_dm)
    minus_dm_smooth = np.zeros_like(minus_dm)
    
    # Initial values (first 14 periods)
    if len(tr) >= 14:
        atr[13] = np.mean(tr[0:14])
        plus_dm_smooth[13] = np.mean(plus_dm[0:14])
        minus_dm_smooth[13] = np.mean(minus_dm[0:14])
        
        # Wilder's smoothing for rest
        for i in range(14, len(tr)):
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
            plus_dm_smooth[i] = (plus_dm_smooth[i-1] * 13 + plus_dm[i]) / 14
            minus_dm_smooth[i] = (minus_dm_smooth[i-1] * 13 + minus_dm[i]) / 14
    
    # Avoid division by zero
    plus_di = np.where(atr != 0, 100 * plus_dm_smooth / atr, 0)
    minus_di = np.where(atr != 0, 100 * minus_dm_smooth / atr, 0)
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    
    # ADX: smoothed DX
    adx = np.zeros_like(dx)
    if len(dx) >= 27:  # Need 14 for initial + 13 more for smoothing
        adx[26] = np.mean(dx[14:27])
        for i in range(27, len(dx)):
            adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    # Volume confirmation: 20-period volume SMA
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = max(20, 27)  # Need Donchian and ADX
    
    for i in range(start_idx, n):
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(adx[i]) or
            np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_sma_val = vol_sma_20[i]
        adx_val = adx[i]
        upper_band = high_20[i]
        lower_band = low_20[i]
        
        if position == 0:
            # Long: break above upper band + volume confirmation + strong trend (ADX > 25)
            if price > upper_band and vol > 1.5 * vol_sma_val and adx_val > 25:
                signals[i] = 0.25
                position = 1
            # Short: break below lower band + volume confirmation + strong trend (ADX > 25)
            elif price < lower_band and vol > 1.5 * vol_sma_val and adx_val > 25:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: break below lower band OR weak trend (ADX < 20)
            if price < lower_band or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: break above upper band OR weak trend (ADX < 20)
            if price > upper_band or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_Volume_ADX"
timeframe = "4h"
leverage = 1.0