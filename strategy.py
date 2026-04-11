#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_trix_volume_regime_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Calculate TRIX on 1d close (15-period)
    close_1d = df_1d['close'].values
    ema1 = pd.Series(close_1d).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix_raw = (ema3 - np.roll(ema3, 1)) / np.roll(ema3, 1)
    trix_raw[0] = 0
    trix = pd.Series(trix_raw).ewm(span=9, adjust=False, min_periods=9).mean().values
    
    # Calculate 1d ADX for regime filter (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
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
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_plus14 = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus14 = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # DI and DX
    di_plus = 100 * dm_plus14 / tr14
    di_minus = 100 * dm_minus14 / tr14
    dx = np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100
    dx = np.where((di_plus + di_minus) == 0, 0, dx)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align 1d indicators to 4h
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation on 4h: volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(200, n):
        # Skip if any required data is invalid
        if (np.isnan(trix_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        trix_val = trix_aligned[i]
        adx_val = adx_aligned[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation
        volume_confirmed = volume_current > 1.8 * vol_ma
        
        # Regime filter: ADX > 25 for trending market
        trending = adx_val > 25
        
        # TRIX signals: zero-cross with momentum
        trix_prev = trix_aligned[i-1] if i > 0 else 0
        
        # Long: TRIX crosses above zero in trending market with volume
        long_signal = (trix_prev <= 0 and trix_val > 0) and trending and volume_confirmed
        
        # Short: TRIX crosses below zero in trending market with volume
        short_signal = (trix_prev >= 0 and trix_val < 0) and trending and volume_confirmed
        
        # Exit when TRIX crosses zero in opposite direction
        exit_long = position == 1 and trix_val < 0
        exit_short = position == -1 and trix_val > 0
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: TRIX (triple exponential average) zero-cross strategy on 4h timeframe.
# Uses 1d TRIX for trend direction and 1d ADX for regime filtering (only trade when ADX > 25).
# Enters long when TRIX crosses above zero in trending market with volume confirmation (>1.8x avg volume).
# Enters short when TRIX crosses below zero in trending market with volume confirmation.
# Exits when TRIX crosses zero in opposite direction.
# Works in both bull and bear markets by trading momentum in the direction of the 1d trend.
# Designed for low trade frequency (target: 20-50 trades/year) to minimize fee drag.