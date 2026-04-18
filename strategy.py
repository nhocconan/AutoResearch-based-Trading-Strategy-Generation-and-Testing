#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 1d ADX regime filter and volume confirmation.
# Williams %R identifies overbought/oversold conditions. ADX > 25 filters for trending markets.
# In trending markets (ADX > 25), we fade extreme Williams %R readings (>80 for short, <20 for long).
# Volume spike (>1.5x 20-period average) confirms the reversal signal.
# Works in both bull and bear markets as it captures mean reversions within trends.
# Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag.
name = "6h_WilliamsR_1dADX25_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX calculation (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Williams %R (14-period)
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    
    # Calculate ADX components (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = pd.Series(high_1d).shift(1) - pd.Series(low_1d)
    tr2 = pd.Series(close_1d).shift(1) - pd.Series(low_1d)
    tr3 = pd.Series(high_1d) - pd.Series(close_1d).shift(1)
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    dm_plus = pd.Series(high_1d).diff()
    dm_minus = pd.Series(low_1d).diff()
    dm_plus = np.where((dm_plus > dm_minus) & (dm_plus > 0), dm_plus, 0)
    dm_minus = np.where((dm_minus > dm_plus) & (dm_minus > 0), dm_minus, 0)
    
    # Smoothed values
    atr_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    # Directional Indicators
    plus_di = 100 * (dm_plus_14 / atr_14)
    minus_di = 100 * (dm_minus_14 / atr_14)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume spike: current volume > 1.5 * 20-period average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(williams_r[i]) or np.isnan(adx_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: ADX > 25 indicates trending market
        trending = adx_aligned[i] > 25
        
        if position == 0:
            # Long: Williams %R oversold (< -80) AND trending AND volume spike
            if williams_r[i] < -80 and trending and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) AND trending AND volume spike
            elif williams_r[i] > -20 and trending and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Williams %R rises above -50 (exit oversold) OR trend weakens
            if williams_r[i] > -50 or not trending:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Williams %R falls below -50 (exit overbought) OR trend weakens
            if williams_r[i] < -50 or not trending:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals