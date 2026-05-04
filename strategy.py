#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R extreme + 1d ADX trend filter + volume confirmation
# Long when Williams %R < -80 (oversold) AND 1d ADX > 25 (trending) AND volume > 1.5x 20 EMA
# Short when Williams %R > -20 (overbought) AND 1d ADX > 25 (trending) AND volume > 1.5x 20 EMA
# Uses 6h for Williams %R signals, 1d for ADX trend to avoid counter-trend trades.
# Works in bull markets via longs in oversold dips and bear markets via shorts in overbought rallies.
# Discrete sizing (0.25) to balance return and fee drag. Target: 12-37 trades/year.

name = "6h_WilliamsR_Extreme_1dADX_Trend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 6h Williams %R (14-period)
    lookback = 14
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero (when highest_high == lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Get 1d data for ADX calculation - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough data for ADX
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX (14-period)
    # True Range
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - np.roll(close_1d, 1)))
    tr3 = pd.Series(np.abs(low_1d - np.roll(close_1d, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).values
    tr[0] = high_1d[0] - low_1d[0]  # First TR
    
    # Directional Movement
    up_move = pd.Series(high_1d - np.roll(high_1d, 1))
    down_move = pd.Series(np.roll(low_1d, 1) - low_1d)
    up_move = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    down_move = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    # Handle first value
    atr[13] = np.mean(tr[0:14])
    
    up_smoothed = pd.Series(up_move).rolling(window=14, min_periods=14).mean().values
    down_smoothed = pd.Series(down_move).rolling(window=14, min_periods=14).mean().values
    up_smoothed[13] = np.mean(up_move[0:14])
    down_smoothed[13] = np.mean(down_move[0:14])
    
    # Directional Indicators
    plus_di = 100 * up_smoothed / atr
    minus_di = 100 * down_smoothed / atr
    # Handle division by zero
    plus_di = np.where(atr == 0, 0, plus_di)
    minus_di = np.where(atr == 0, 0, minus_di)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = np.where((plus_di + minus_di) == 0, 0, dx)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx[27] = np.mean(dx[14:28])  # First ADX value
    
    # Align 1d ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Williams %R thresholds
    williams_oversold = williams_r < -80
    williams_overbought = williams_r > -20
    
    # ADX trend filter (trending when ADX > 25)
    adx_trending = adx_aligned > 25
    
    # Volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(williams_r[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Williams %R oversold AND ADX trending AND volume spike
            if (williams_oversold[i] and 
                adx_trending[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Williams %R overbought AND ADX trending AND volume spike
            elif (williams_overbought[i] and 
                  adx_trending[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R rises above -50 OR ADX loses trend
            if (williams_r[i] > -50 or 
                adx_aligned[i] <= 25):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R falls below -50 OR ADX loses trend
            if (williams_r[i] < -50 or 
                adx_aligned[i] <= 25):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals