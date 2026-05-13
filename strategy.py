#!/usr/bin/env python3
# Hypothesis: 12h Williams %R Mean Reversion with 1d ADX regime filter and volume confirmation.
# Long when Williams %R < -80 (oversold), 1d ADX < 25 (range market), and volume > 1.5x 20-bar average.
# Short when Williams %R > -20 (overbought), 1d ADX < 25 (range market), and volume > 1.5x 20-bar average.
# Uses discrete sizing 0.25 to target 50-150 total trades over 4 years on 12h timeframe.
# Williams %R identifies overextended moves in range markets; ADX filter avoids trending markets where mean reversion fails.
# Works in bull markets via mean reversion at extremes and in bear markets via the same mechanism during range-bound periods.

name = "12h_WilliamsR_MeanReversion_1dADX25_VolumeConfirm"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    lookback = 20  # for volume average and Williams %R
    
    # Get 1d data for Williams %R and ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < lookback + 1:  # Need enough data for indicators
        return np.zeros(n)
    
    # Calculate Williams %R on 1d: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(df_1d['high']).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(df_1d['low']).rolling(window=lookback, min_periods=lookback).min().values
    williams_r = (highest_high - df_1d['close'].values) / (highest_high - lowest_low) * -100
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Align Williams %R to 12h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Calculate ADX on 1d for regime filter
    # True Range
    tr1 = pd.Series(df_1d['high']).diff().abs()
    tr2 = (pd.Series(df_1d['high']) - pd.Series(df_1d['low'].shift(1))).abs()
    tr3 = (pd.Series(df_1d['low']) - pd.Series(df_1d['close'].shift(1))).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).values
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    up_move = pd.Series(df_1d['high']).diff()
    down_move = -pd.Series(df_1d['low']).diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM and TR
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values
    atr_smooth = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr_smooth
    minus_di = 100 * minus_dm_smooth / atr_smooth
    # Avoid division by zero
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=lookback, min_periods=lookback).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback, n):  # Start after sufficient data
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Williams %R < -80 (oversold), ADX < 25 (range), volume spike
            if (williams_r_aligned[i] < -80 and 
                adx_aligned[i] < 25 and 
                volume[i] > 1.5 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Williams %R > -20 (overbought), ADX < 25 (range), volume spike
            elif (williams_r_aligned[i] > -20 and 
                  adx_aligned[i] < 25 and 
                  volume[i] > 1.5 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Williams %R > -50 (return to mean) OR ADX > 30 (trend emerging)
            if (williams_r_aligned[i] > -50 or 
                adx_aligned[i] > 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Williams %R < -50 (return to mean) OR ADX > 30 (trend emerging)
            if (williams_r_aligned[i] < -50 or 
                adx_aligned[i] > 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals