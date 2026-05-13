#!/usr/bin/env python3
# Hypothesis: 12h Williams %R mean reversion with 1d ADX > 20 trend filter and volume spike > 1.8x average.
# Long when Williams %R < -80 (oversold) with 1d ADX > 20 and volume > 1.8x 20-bar average volume.
# Short when Williams %R > -20 (overbought) with 1d ADX > 20 and volume > 1.8x average.
# Exit when Williams %R crosses above -50 for longs or below -50 for shorts.
# Uses discrete position sizing 0.25. Target: 50-150 total trades over 4 years on 12h timeframe.
# 1d ADX ensures we only trade in trending markets, avoiding choppy conditions that cause false signals.
# Williams %R provides mean reversion entries within the trend, suitable for 12h timeframe.

name = "12h_WilliamsR_MeanReversion_1dADX20_VolumeSpike_v1"
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
    
    # Calculate Williams %R (14-period)
    lookback = 14
    if n < lookback + 1:
        return np.zeros(n)
    
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data for ADX20 trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX on 1d data
    # True Range
    tr1 = np.abs(np.diff(high_1d, prepend=high_1d[0]))
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    up_move = np.diff(high_1d, prepend=high_1d[0])
    down_move = -(np.diff(low_1d, prepend=low_1d[0]))
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, min_periods=14, adjust=False).mean() / atr_1d
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, min_periods=14, adjust=False).mean() / atr_1d
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    # Handle division by zero when plus_di + minus_di == 0
    dx = np.where((plus_di + minus_di) == 0, 0, dx)
    adx_1d = pd.Series(dx).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    
    # Align 1d ADX to 12h timeframe (wait for 1d bar to close)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback + 20, n):  # Start after sufficient data
        # Skip if any required data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Williams %R < -80 (oversold) with 1d ADX > 20 and volume spike > 1.8x
            if (williams_r[i] < -80 and 
                adx_1d_aligned[i] > 20 and 
                volume[i] > 1.8 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Williams %R > -20 (overbought) with 1d ADX > 20 and volume spike > 1.8x
            elif (williams_r[i] > -20 and 
                  adx_1d_aligned[i] > 20 and 
                  volume[i] > 1.8 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Williams %R crosses above -50
            if williams_r[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Williams %R crosses below -50
            if williams_r[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals