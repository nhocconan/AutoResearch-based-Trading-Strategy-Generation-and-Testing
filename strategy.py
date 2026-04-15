#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams %R with 1w ADX trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions, effective in both bull and bear markets.
# 1w ADX ensures we only trade in trending markets (ADX > 25), avoiding whipsaws in ranges.
# Volume confirms the strength of the reversal. Target: 20-60 trades over 4 years (5-15/year).
# Uses daily timeframe for signals with weekly trend filter to reduce noise and improve win rate.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data (primary timeframe) for Williams %R
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Load 1w data for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Williams %R (14-period) on 1d
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14 + 1e-10)
    
    # Calculate ADX (14-period) on 1w
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    up_move = np.diff(high_1w, prepend=high_1w[0])
    down_move = -np.diff(low_1w, prepend=low_1w[0])  # negative of downward move
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed TR, +DM, -DM
    atr_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / (atr_1w + 1e-10)
    minus_di = 100 * minus_dm_smooth / (atr_1w + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Volume average (20-period on 1d)
    vol_avg_1d = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to 1d timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(adx_aligned[i]) or
            np.isnan(vol_avg_aligned[i])):
            continue
        
        # Long entry: Williams %R crosses above -80 from oversold + ADX > 25 (trending) + volume spike
        if (williams_r_aligned[i] > -80 and williams_r_aligned[i-1] <= -80 and
            adx_aligned[i] > 25 and
            volume[i] > 1.5 * vol_avg_aligned[i] and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: Williams %R crosses below -20 from overbought + ADX > 25 (trending) + volume spike
        elif (williams_r_aligned[i] < -20 and williams_r_aligned[i-1] >= -20 and
              adx_aligned[i] > 25 and
              volume[i] > 1.5 * vol_avg_aligned[i] and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse Williams %R signal or ADX < 20 (ranging market)
        elif position == 1 and (williams_r_aligned[i] < -20 or adx_aligned[i] < 20):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (williams_r_aligned[i] > -80 or adx_aligned[i] < 20):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "1d_WilliamsR_ADX_Volume_Filter"
timeframe = "1d"
leverage = 1.0