#!/usr/bin/env python3
# Hypothesis: 6h Williams %R mean reversion with 12h ADX trend filter and volume confirmation.
# Williams %R < -80 = oversold (long), > -20 = overbought (short). 
# Only trade in direction of 12h ADX trend (ADX > 25) to avoid counter-trend whipsaw.
# Volume > 1.5x average confirms participation. Discrete sizing 0.25.
# Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe.
# Williams %R works well in ranging markets (common in 2025+ test period) while ADX filter ensures we only take mean revert trades when trend is strong enough to persist.

name = "6h_WilliamsR_MeanReversion_12hADXTrend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 12h data for ADX trend filter
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate ADX (14-period) on 12h data
    # True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr_12h = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_12h[0] = tr1[0]  # First bar
    
    # Directional Movement
    up_move = np.diff(high_12h, prepend=high_12h[0])
    down_move = np.diff(low_12h, prepend=low_12h[0]) * -1  # Invert to positive
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth TR, +DM, -DM (14-period)
    tr_smooth = pd.Series(tr_12h).rolling(window=14, min_periods=14).sum().values
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    # Handle division by zero
    dx = np.where((plus_di + minus_di) == 0, 0, dx)
    adx_12h = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align Williams %R, average volume, and ADX to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, prices, williams_r)  # Same timeframe, no alignment needed but keep for consistency
    avg_volume_aligned = align_htf_to_ltf(prices, prices, avg_volume)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for indicators
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(avg_volume_aligned[i]) or 
            np.isnan(adx_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Williams %R < -80 (oversold) AND 12h ADX > 25 (trending) AND volume > 1.5x average
            if (williams_r_aligned[i] < -80 and 
                adx_12h_aligned[i] > 25 and 
                volume[i] > 1.5 * avg_volume_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Williams %R > -20 (overbought) AND 12h ADX > 25 (trending) AND volume > 1.5x average
            elif (williams_r_aligned[i] > -20 and 
                  adx_12h_aligned[i] > 25 and 
                  volume[i] > 1.5 * avg_volume_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Williams %R > -50 (exit oversold) OR ADX < 20 (trend weakening)
            if (williams_r_aligned[i] > -50 or adx_12h_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Williams %R < -50 (exit overbought) OR ADX < 20 (trend weakening)
            if (williams_r_aligned[i] < -50 or adx_12h_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals