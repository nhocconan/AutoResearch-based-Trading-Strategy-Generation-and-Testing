#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R (14) combined with 1-day ADX trend filter and volume confirmation.
# Williams %R identifies overbought/oversold conditions; ADX confirms trend strength; volume ensures momentum.
# In trending markets (ADX > 25), we fade extreme %R readings for mean reversion.
# In ranging markets (ADX < 20), we avoid trades to prevent whipsaw.
# Designed for 6h timeframe with ~15-35 trades/year to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for ADX calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX(14) on daily data
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (alpha=1/14)
    atr_1d = np.full_like(tr, np.nan)
    plus_dm_smooth = np.full_like(plus_dm, np.nan)
    minus_dm_smooth = np.full_like(minus_dm, np.nan)
    
    # Initialize first values
    atr_1d[13] = np.mean(tr[1:14])  # 14-period average
    plus_dm_smooth[13] = np.mean(plus_dm[1:14])
    minus_dm_smooth[13] = np.mean(minus_dm[1:14])
    
    # Wilder's smoothing
    for i in range(14, len(tr)):
        atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
        plus_dm_smooth[i] = (plus_dm_smooth[i-1] * 13 + plus_dm[i]) / 14
        minus_dm_smooth[i] = (minus_dm_smooth[i-1] * 13 + minus_dm[i]) / 14
    
    # Directional Indicators
    plus_di_1d = np.full_like(close_1d, np.nan)
    minus_di_1d = np.full_like(close_1d, np.nan)
    dx_1d = np.full_like(close_1d, np.nan)
    
    for i in range(13, len(tr)):
        if atr_1d[i] != 0:
            plus_di_1d[i] = 100 * plus_dm_smooth[i] / atr_1d[i]
            minus_di_1d[i] = 100 * minus_dm_smooth[i] / atr_1d[i]
            if (plus_di_1d[i] + minus_di_1d[i]) != 0:
                dx_1d[i] = 100 * np.abs(plus_di_1d[i] - minus_di_1d[i]) / (plus_di_1d[i] + minus_di_1d[i])
    
    # ADX: smoothed DX
    adx_1d = np.full_like(close_1d, np.nan)
    dx_valid = dx_1d[~np.isnan(dx_1d)]
    if len(dx_valid) >= 14:
        adx_1d[26] = np.mean(dx_1d[13:27])  # First ADX after 14 DX periods
        for i in range(27, len(dx_1d)):
            if not np.isnan(dx_1d[i]):
                adx_1d[i] = (adx_1d[i-1] * 13 + dx_1d[i]) / 14
    
    # Align daily ADX to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate Williams %R on 6h data
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(13, n):
        highest_high[i] = np.max(high[i-13:i+1])
        lowest_low[i] = np.min(low[i-13:i+1])
    
    williams_r = np.full(n, np.nan)
    for i in range(13, n):
        if highest_high[i] != lowest_low[i]:
            williams_r[i] = -100 * (highest_high[i] - close[i]) / (highest_high[i] - lowest_low[i])
    
    # Volume confirmation: 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(27, 20)  # need ADX, Williams %R, volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3 * 20-period average
        vol_confirmed = volume[i] > 1.3 * vol_ma[i]
        
        # Regime filter: ADX > 25 for trending, ADX < 20 for ranging
        is_trending = adx_1d_aligned[i] > 25
        is_ranging = adx_1d_aligned[i] < 20
        
        if position == 0:
            # Long entry: Williams %R oversold (< -80) in trending market with volume
            if (williams_r[i] < -80 and 
                is_trending and 
                vol_confirmed):
                signals[i] = 0.25
                position = 1
            # Short entry: Williams %R overbought (> -20) in trending market with volume
            elif (williams_r[i] > -20 and 
                  is_trending and 
                  vol_confirmed):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: Williams %R returns to midpoint (-50) or overbought
            if williams_r[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Williams %R returns to midpoint (-50) or oversold
            if williams_r[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_ADXTrend_VolumeFilter"
timeframe = "6h"
leverage = 1.0