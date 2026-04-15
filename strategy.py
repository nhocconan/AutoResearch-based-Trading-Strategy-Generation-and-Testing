#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h ADX Trend Strength with 1d Momentum Confirmation
# Uses ADX > 25 to identify trending markets on 4h, then enters in the direction of
# 1-day price momentum (close vs open). Long when 1d momentum positive, short when negative.
# Includes volume confirmation to avoid false breakouts. Designed to work in both bull and bear markets
# by capturing strong trends regardless of direction. Target: 50-150 total trades over 4 years.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data for ADX calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Load 1d data for momentum
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    open_1d = df_1d['open'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX (14-period) on 4h
    # True Range
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    dm_plus = np.where((high_4h - np.roll(high_4h, 1)) > (np.roll(low_4h, 1) - low_4h), 
                       np.maximum(high_4h - np.roll(high_4h, 1), 0), 0)
    dm_minus = np.where((np.roll(low_4h, 1) - low_4h) > (high_4h - np.roll(high_4h, 1)), 
                        np.maximum(np.roll(low_4h, 1) - low_4h, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / (atr + 1e-10)
    di_minus = 100 * dm_minus_smooth / (atr + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1-day momentum (close - open)
    momentum_1d = close_1d - open_1d
    
    # Align ADX and momentum to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_4h, adx)
    momentum_1d_aligned = align_htf_to_ltf(prices, df_1d, momentum_1d)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_aligned[i]) or np.isnan(momentum_1d_aligned[i])):
            continue
        
        # Long entry: ADX > 25 (trending) + positive 1-day momentum + volume confirmation
        if (adx_aligned[i] > 25 and
            momentum_1d_aligned[i] > 0 and
            volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: ADX > 25 (trending) + negative 1-day momentum + volume confirmation
        elif (adx_aligned[i] > 25 and
              momentum_1d_aligned[i] < 0 and
              volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: ADX < 20 (ranging market) or momentum reversal
        elif position == 1 and (adx_aligned[i] < 20 or momentum_1d_aligned[i] < 0):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (adx_aligned[i] < 20 or momentum_1d_aligned[i] > 0):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "4h_ADX_Trend_1d_Momentum"
timeframe = "4h"
leverage = 1.0