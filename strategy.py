#!/usr/bin/env python3
# Hypothesis: 6h Williams %R extreme reversal with 1d ADX trend filter and volume confirmation.
# Long when Williams %R < -80 (oversold) AND 1d ADX > 25 (trending) AND volume > 1.5x average
# Short when Williams %R > -20 (overbought) AND 1d ADX > 25 (trending) AND volume > 1.5x average
# Exit when Williams %R crosses -50 (mean reversion) OR ADX < 20 (trend weakness)
# Uses 6h timeframe for lower frequency, Williams %R for momentum extremes, 1d ADX for trend filter, volume for confirmation.
# Target: 50-150 total trades over 4 years (12-37/year). Works in bull via trend continuation, bear via oversold bounces.

name = "6h_WilliamsR_ADX_Trend_Volume_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data for Williams %R calculation
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Calculate Williams %R(14) on 6h data
    if len(high_6h) >= 14:
        highest_high = pd.Series(high_6h).rolling(window=14, min_periods=14).max().values
        lowest_low = pd.Series(low_6h).rolling(window=14, min_periods=14).min().values
        williams_r = -100 * (highest_high - close_6h) / (highest_high - lowest_low)
        # Handle division by zero when highest_high == lowest_low
        williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    else:
        williams_r = np.full_like(close_6h, np.nan)
    
    # Align Williams %R to 6h timeframe (already aligned since calculated on 6h)
    williams_r_aligned = align_htf_to_ltf(prices, df_6h, williams_r)
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX(14) on 1d data
    if len(high_1d) >= 14:
        # True Range
        tr1 = pd.Series(high_1d).diff().abs()
        tr2 = (pd.Series(high_1d) - pd.Series(close_1d).shift()).abs()
        tr3 = (pd.Series(low_1d) - pd.Series(close_1d).shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
        
        # Directional Movement
        up_move = pd.Series(high_1d).diff()
        down_move = -pd.Series(low_1d).diff()
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed DM
        plus_dm_smooth = pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
        minus_dm_smooth = pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
        
        # Directional Indicators
        plus_di = 100 * plus_dm_smooth / atr
        minus_di = 100 * minus_dm_smooth / atr
        
        # DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        dx = np.where((plus_di + minus_di) == 0, 0, dx)
        adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    else:
        adx = np.full_like(close_1d, np.nan)
    
    # Align ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume filter: current 6h volume > 1.5x 20-period average (spike confirmation)
    vol_ma_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_6h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after sufficient data for Williams %R and ADX
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(adx_aligned[i]) or np.isnan(vol_ma_6h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Williams %R < -80 (oversold) AND ADX > 25 (trending) AND volume spike
            if williams_r_aligned[i] < -80 and adx_aligned[i] > 25 and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Williams %R > -20 (overbought) AND ADX > 25 (trending) AND volume spike
            elif williams_r_aligned[i] > -20 and adx_aligned[i] > 25 and volume_filter[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Williams %R > -50 (mean reversion) OR ADX < 20 (trend weakness)
            if williams_r_aligned[i] > -50 or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Williams %R < -50 (mean reversion) OR ADX < 20 (trend weakness)
            if williams_r_aligned[i] < -50 or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals