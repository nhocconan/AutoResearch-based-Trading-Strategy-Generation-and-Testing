#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(15) breakout with 1d volume confirmation and 1d ADX trend filter
# Uses Donchian channel breakouts for trend capture, volume to confirm breakout strength,
# and ADX to avoid ranging markets. Works in both bull and bear by
# only taking breakouts when ADX > 25 (trending market).
# Target: 50-150 total trades over 4 years (12-37/year) with disciplined entries.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data (primary timeframe) for price action
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Load 1d data for volume and ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    vol_1d = df_1d['volume'].values
    
    # Calculate Donchian channels (15-period) on 12h
    donch_high_12h = pd.Series(high_12h).rolling(window=15, min_periods=15).max().values
    donch_low_12h = pd.Series(low_12h).rolling(window=15, min_periods=15).min().values
    
    # Calculate ADX (14-period) on 1d
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed values
    atr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    plus_di_14 = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean().values / (atr_14 + 1e-10)
    minus_di_14 = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean().values / (atr_14 + 1e-10)
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14 + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    
    # Volume average (20-period on 1d)
    vol_avg_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 12h timeframe
    donch_high_12h_aligned = align_htf_to_ltf(prices, df_12h, donch_high_12h)
    donch_low_12h_aligned = align_htf_to_ltf(prices, df_12h, donch_low_12h)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_high_12h_aligned[i]) or np.isnan(donch_low_12h_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_avg_aligned[i])):
            continue
        
        # Long entry: price breaks above Donchian high + volume spike + ADX > 25 (trending)
        if (close[i] > donch_high_12h_aligned[i] and
            volume[i] > 1.5 * vol_avg_aligned[i] and
            adx_aligned[i] > 25 and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price breaks below Donchian low + volume spike + ADX > 25 (trending)
        elif (close[i] < donch_low_12h_aligned[i] and
              volume[i] > 1.5 * vol_avg_aligned[i] and
              adx_aligned[i] > 25 and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse signal or ADX < 20 (ranging market) - hysteresis to prevent whipsaw
        elif position == 1 and (close[i] < donch_low_12h_aligned[i] or adx_aligned[i] < 20):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > donch_high_12h_aligned[i] or adx_aligned[i] < 20):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "12h_Donchian_Volume_ADX_Filter"
timeframe = "12h"
leverage = 1.0