#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + volume confirmation + 1d ADX trend filter
# Uses Donchian channel breakouts for trend capture, volume to confirm breakout strength,
# and ADX to filter ranging markets. Works in both bull and bear by
# only taking breakouts in the direction of the 1d trend (EMA50).
# Target: 100-200 total trades over 4 years (25-50/year) with disciplined entries.

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
    
    # Load 1d data for ADX and EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Donchian channels (20-period) on 12h
    donch_high_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donch_low_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Calculate ADX (14-period) on 1d
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    atr_14_smooth = pd.Series(atr_14).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / (atr_14_smooth + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / (atr_14_smooth + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Calculate EMA50 on 1d for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume average (20-period on 12h)
    vol_avg_12h = pd.Series(df_12h['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 12h timeframe
    donch_high_12h_aligned = align_htf_to_ltf(prices, df_12h, donch_high_12h)
    donch_low_12h_aligned = align_htf_to_ltf(prices, df_12h, donch_low_12h)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    vol_avg_aligned = align_htf_to_ltf(prices, df_12h, vol_avg_12h)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_high_12h_aligned[i]) or np.isnan(donch_low_12h_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or
            np.isnan(vol_avg_aligned[i])):
            continue
        
        # Long entry: price breaks above Donchian high + volume spike + ADX > 25 (trending) + price above EMA50
        if (close[i] > donch_high_12h_aligned[i] and
            volume[i] > 1.5 * vol_avg_aligned[i] and
            adx_aligned[i] > 25 and
            close[i] > ema50_1d_aligned[i] and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price breaks below Donchian low + volume spike + ADX > 25 (trending) + price below EMA50
        elif (close[i] < donch_low_12h_aligned[i] and
              volume[i] > 1.5 * vol_avg_aligned[i] and
              adx_aligned[i] > 25 and
              close[i] < ema50_1d_aligned[i] and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse signal or ADX < 20 (ranging market)
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