#!/usr/bin/env python3
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
    
    # Get daily data for ADX calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX(14) on daily data
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high_1d[0] - low_1d[0]  # First value
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = np.where((plus_di + minus_di) == 0, 0, dx)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Get 6h EMA34 for trend filter
    ema34 = pd.Series(close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume filter: current volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(adx_aligned[i]) or np.isnan(ema34[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: ADX > 25 (trending) + price > EMA34 + volume confirmation
            if adx_aligned[i] > 25 and close[i] > ema34[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: ADX > 25 (trending) + price < EMA34 + volume confirmation
            elif adx_aligned[i] > 25 and close[i] < ema34[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: ADX < 20 (no trend) OR price < EMA34
            if adx_aligned[i] < 20 or close[i] < ema34[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: ADX < 20 (no trend) OR price > EMA34
            if adx_aligned[i] < 20 or close[i] > ema34[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ADX25_EMA34_VolumeFilter"
timeframe = "6h"
leverage = 1.0