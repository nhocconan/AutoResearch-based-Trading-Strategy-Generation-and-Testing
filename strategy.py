#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R extreme reversal with 1d ADX trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions (-20 oversold, -80 overbought)
# ADX > 25 ensures we only trade in trending markets, avoiding whipsaws in ranges
# Volume confirmation ensures breakout participation
# Discrete position sizing 0.25 balances risk and minimizes fee churn
# Targets 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits
# Works in both bull and bear markets by only taking reversals in established trends

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
    
    # Load 1d data ONCE before loop for ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX(14) for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with close_1d index
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing)
    tr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align ADX to 6h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 6h Williams %R(14)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Calculate 6h volume confirmation (1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Williams %R and volume MA)
    start_idx = 34  # max(14 for Williams %R, 20 for volume) + buffer
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(williams_r[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R < -80 (oversold) AND ADX > 25 (trending) AND volume confirm
            if (williams_r[i] < -80 and 
                adx_aligned[i] > 25 and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought) AND ADX > 25 (trending) AND volume confirm
            elif (williams_r[i] > -20 and 
                  adx_aligned[i] > 25 and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R > -20 (overbought) OR ADX < 20 (trend weakening)
            if (williams_r[i] > -20 or adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R < -80 (oversold) OR ADX < 20 (trend weakening)
            if (williams_r[i] < -80 or adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals