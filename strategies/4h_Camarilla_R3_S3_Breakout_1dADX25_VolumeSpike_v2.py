#!/usr/bin/env python3
# Hypothesis: 4h Camarilla R3/S3 breakout with 1d ADX > 25 trend filter and volume spike > 2.0x average.
# Long when price closes above R3 with 1d ADX > 25 and volume > 2.0x 20-bar average volume.
# Short when price closes below S3 with 1d ADX > 25 and volume > 2.0x average.
# Exit when price reverses and closes below/above the opposite Camarilla level (S3 for longs, R3 for shorts).
# Uses discrete position sizing 0.25. Target: 75-200 total trades over 4 years on 4h timeframe.
# Higher volume threshold (2.0x vs 1.8x) reduces overtrading and fee drag while maintaining edge in strong moves.
# 1d ADX ensures we only trade in trending markets, avoiding choppy conditions that cause false breakouts.

name = "4h_Camarilla_R3_S3_Breakout_1dADX25_VolumeSpike_v2"
timeframe = "4h"
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
    
    # Calculate Camarilla levels from previous day (approx using 6x 4h bars)
    lookback = 6  # 6 * 4h = 24h approx
    if n < lookback + 1:
        return np.zeros(n)
    
    # Calculate rolling max/min/close for previous "day"
    high_prev = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().shift(1).values
    low_prev = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().shift(1).values
    close_prev = pd.Series(close).rolling(window=lookback, min_periods=lookback).mean().shift(1).values
    
    # Camarilla R3 and S3 levels
    camarilla_range = high_prev - low_prev
    r3 = close_prev + 1.1 * camarilla_range / 2
    s3 = close_prev - 1.1 * camarilla_range / 2
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data for ADX25 trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX on 1d data
    # True Range
    tr1 = pd.Series(high_1d).diff().abs()
    tr2 = (pd.Series(high_1d) - pd.Series(close_1d).shift(1)).abs()
    tr3 = (pd.Series(low_1d) - pd.Series(close_1d).shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    up_move = pd.Series(high_1d).diff()
    down_move = -(pd.Series(low_1d).diff())
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, min_periods=14, adjust=False).mean() / atr_1d
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, min_periods=14, adjust=False).mean() / atr_1d
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_1d = pd.Series(dx).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    
    # Align 1d ADX to 4h timeframe (wait for 1d bar to close)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback + 20, n):  # Start after sufficient data
        # Skip if any required data is NaN
        if (np.isnan(r3[i]) or np.isnan(s3[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price closes above R3 with 1d ADX > 25 and volume spike > 2.0x
            if (close[i] > r3[i] and 
                adx_1d_aligned[i] > 25 and 
                volume[i] > 2.0 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price closes below S3 with 1d ADX > 25 and volume spike > 2.0x
            elif (close[i] < s3[i] and 
                  adx_1d_aligned[i] > 25 and 
                  volume[i] > 2.0 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below S3 (reversal signal)
            if close[i] < s3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above R3 (reversal signal)
            if close[i] > r3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals