#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R Extreme + 1d ADX Trend Filter + Volume Confirmation
# Williams %R(14) identifies overbought/oversold conditions: < -80 = oversold, > -20 = overbought
# 1d ADX(14) > 25 filters for trending markets to avoid whipsaws in ranging conditions
# Volume spike (1.5x 20-period average) confirms participation on signals
# Only takes long when %R < -80 AND ADX > 25 AND volume confirms
# Only takes short when %R > -20 AND ADX > 25 AND volume confirms
# Discrete position sizing 0.25 balances risk and minimizes fee churn
# Targets 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits
# Works in both bull and bear markets by requiring strong trend (ADX>25) for entries

name = "6h_WilliamsR_EXTREME_1dADX_VolumeConfirm_v1"
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
    
    # Load 1d data ONCE before loop for ADX and Williams %R calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Williams %R(14): (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    close_1d = df_1d['close'].values
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    
    # Calculate 1d ADX(14)
    # True Range
    tr1 = pd.Series(df_1d['high']).values - pd.Series(df_1d['low']).values
    tr2 = np.abs(pd.Series(df_1d['high']).values - pd.Series(df_1d['close']).shift(1).values)
    tr3 = np.abs(pd.Series(df_1d['low']).values - pd.Series(df_1d['close']).shift(1).values)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    dm_plus = np.where(
        (pd.Series(df_1d['high']).values - pd.Series(df_1d['high']).shift(1).values) >
        (pd.Series(df_1d['low']).shift(1).values - pd.Series(df_1d['low']).values),
        np.maximum(pd.Series(df_1d['high']).values - pd.Series(df_1d['high']).shift(1).values, 0),
        0
    )
    dm_minus = np.where(
        (pd.Series(df_1d['low']).shift(1).values - pd.Series(df_1d['low']).values) >
        (pd.Series(df_1d['high']).values - pd.Series(df_1d['high']).shift(1).values),
        np.maximum(pd.Series(df_1d['low']).shift(1).values - pd.Series(df_1d['low']).values, 0),
        0
    )
    
    # Smoothed DM and TR
    dm_plus_smooth = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    tr_smooth = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / tr_smooth
    di_minus = 100 * dm_minus_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align Williams %R and ADX to 6h
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 6h volume confirmation (1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Williams %R, ADX, and volume MA)
    start_idx = 50  # max(14 for Williams %R/ADX, 20 for volume) + buffer
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(williams_r_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend strength from 1d ADX
        strong_trend = adx_aligned[i] > 25
        
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R < -80 (oversold) AND strong trend AND volume confirm
            if (williams_r_aligned[i] < -80 and 
                strong_trend and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought) AND strong trend AND volume confirm
            elif (williams_r_aligned[i] > -20 and 
                  strong_trend and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R > -50 (recovering from oversold) OR trend weakens
            if (williams_r_aligned[i] > -50 or 
                not strong_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R < -50 (declining from overbought) OR trend weakens
            if (williams_r_aligned[i] < -50 or 
                not strong_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals