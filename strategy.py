#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d ADX(14) trend filter + volume confirmation
# Uses 12h primary timeframe for Donchian breakout signals (long on upper band, short on lower band)
# 1d ADX > 25 confirms strong trend (avoids ranging markets)
# Volume confirmation (1.5x 20-period average) ensures strong participation
# Discrete position sizing (0.25) balances profit potential with fee drag minimization
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# Donchian provides clear structure, ADX filters choppy conditions, volume confirms conviction
# Works in both bull and bear markets by only trading when trend is strong

name = "12h_Donchian20_1dADX25_Trend_Volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX (Average Directional Index)
    high_1d = pd.Series(df_1d['high'])
    low_1d = pd.Series(df_1d['low'])
    close_1d = pd.Series(df_1d['close'])
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = abs(high_1d - close_1d.shift(1))
    tr3 = abs(low_1d - close_1d.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean()
    
    # Directional Movement
    dm_plus = np.where((high_1d - high_1d.shift(1)) > (low_1d.shift(1) - low_1d), 
                       np.maximum(high_1d - high_1d.shift(1), 0), 0)
    dm_minus = np.where((low_1d.shift(1) - low_1d) > (high_1d - high_1d.shift(1)), 
                        np.maximum(low_1d.shift(1) - low_1d, 0), 0)
    
    # Smoothed DM and TR
    dm_plus_smooth = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum()
    dm_minus_smooth = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum()
    tr_smooth = pd.Series(tr).rolling(window=14, min_periods=14).sum()
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / tr_smooth
    di_minus = 100 * dm_minus_smooth / tr_smooth
    
    # ADX
    dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 12h Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation (1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Donchian and ADX calculations)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Donchian breakout: price breaks above upper band (long) or below lower band (short)
            # ADX > 25 confirms strong trend
            # Volume spike ensures strong participation
            if close[i] > donchian_upper[i] and adx_aligned[i] > 25 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            elif close[i] < donchian_lower[i] and adx_aligned[i] > 25 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price breaks below Donchian lower band or ADX weakens (< 20)
            if close[i] < donchian_lower[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian upper band or ADX weakens (< 20)
            if close[i] > donchian_upper[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals