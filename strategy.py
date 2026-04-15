#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R + volume spike + 1d ADX trend filter
# Williams %R identifies overbought/oversold conditions with mean-reversion potential.
# Volume spike confirms momentum behind the move.
# 1d ADX > 25 ensures we only trade in trending markets, avoiding chop.
# Works in both bull and bear by taking mean-reversion entries only when the higher timeframe is trending.
# Target: 60-120 total trades over 4 years (15-30/year) with disciplined entries.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data (primary timeframe) for Williams %R and price action
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Load 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams %R (14-period) on 4h
    highest_high = pd.Series(high_4h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_4h).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_4h) / (highest_high - lowest_low + 1e-10)
    
    # Calculate ADX (14-period) on 1d for trend filter
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    
    # Smooth TR, DM+, DM- using Wilder's smoothing (equivalent to EMA with alpha=1/14)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / (atr + 1e-10)
    di_minus = 100 * dm_minus_smooth / (atr + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Volume average (20-period on 4h)
    vol_avg_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 4h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_4h, williams_r)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    vol_avg_aligned = align_htf_to_ltf(prices, df_4h, vol_avg_4h)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(adx_aligned[i]) or
            np.isnan(vol_avg_aligned[i])):
            continue
        
        # Long entry: Williams %R crosses above -80 (oversold) + volume spike + ADX > 25 (trending)
        if (williams_r_aligned[i] > -80 and williams_r_aligned[i-1] <= -80 and
            volume[i] > 1.5 * vol_avg_aligned[i] and
            adx_aligned[i] > 25 and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: Williams %R crosses below -20 (overbought) + volume spike + ADX > 25 (trending)
        elif (williams_r_aligned[i] < -20 and williams_r_aligned[i-1] >= -20 and
              volume[i] > 1.5 * vol_avg_aligned[i] and
              adx_aligned[i] > 25 and
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

name = "4h_WilliamsR_Volume_ADX_Filter"
timeframe = "4h"
leverage = 1.0