#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R extreme + 1d ADX25 trend filter + volume spike confirmation
# Long when Williams %R < -80 (oversold) AND 1d ADX > 25 (trending) AND volume > 2.0x 20-period average
# Short when Williams %R > -20 (overbought) AND 1d ADX > 25 (trending) AND volume > 2.0x 20-period average
# Exit when Williams %R crosses above -50 (for longs) or below -50 (for shorts) OR price retouches 1d EMA34 (mean reversion)
# Uses 6h primary timeframe with 1d HTF for Williams %R, ADX, and EMA34
# Williams %R identifies extreme reversals in both bull and bear markets
# ADX filter ensures we only trade in trending conditions, reducing whipsaw
# Volume spike confirmation filters low-momentum signals
# Discrete sizing (0.25) to limit fee drag and manage drawdown
# Target: 75-150 total trades over 4 years (19-37/year) for 6h timeframe

name = "6h_WilliamsR_Extreme_1dADX25_Trend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for all indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for exit condition
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d Williams %R (14-period)
    if len(df_1d) >= 14:
        highest_high = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
        lowest_low = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
        williams_r = -100 * (highest_high - df_1d['close'].values) / (highest_high - lowest_low)
        # Handle division by zero
        williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
        williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    else:
        williams_r_aligned = np.full(n, np.nan)
    
    # Calculate 1d ADX (14-period)
    if len(df_1d) >= 14:
        # True Range
        tr1 = pd.Series(df_1d['high'].values - df_1d['low'].values)
        tr2 = pd.Series(abs(df_1d['high'].values - df_1d['close'].shift(1).values))
        tr3 = pd.Series(abs(df_1d['low'].values - df_1d['close'].shift(1).values))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).values
        atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
        
        # Directional Movement
        up_move = pd.Series(df_1d['high'].values - df_1d['high'].shift(1).values)
        down_move = pd.Series(df_1d['low'].shift(1).values - df_1d['low'].values)
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        # Smoothed DM
        plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
        minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
        
        # DI and DX
        plus_di = 100 * plus_dm_smooth / atr
        minus_di = 100 * minus_dm_smooth / atr
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        # Handle division by zero
        dx = np.where((plus_di + minus_di) == 0, 0, dx)
        adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
        adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    else:
        adx_aligned = np.full(n, np.nan)
    
    # Volume confirmation: volume > 2.0x 20-period average
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(adx_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Williams %R < -80 (oversold) AND ADX > 25 (trending) AND volume spike
            if (williams_r_aligned[i] < -80 and 
                adx_aligned[i] > 25 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Williams %R > -20 (overbought) AND ADX > 25 (trending) AND volume spike
            elif (williams_r_aligned[i] > -20 and 
                  adx_aligned[i] > 25 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R crosses above -50 OR price retouches 1d EMA34 (mean reversion)
            if williams_r_aligned[i] > -50 or abs(close[i] - ema_34_1d_aligned[i]) < 0.001 * ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R crosses below -50 OR price retouches 1d EMA34 (mean reversion)
            if williams_r_aligned[i] < -50 or abs(close[i] - ema_34_1d_aligned[i]) < 0.001 * ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals