#!/usr/bin/env python3
"""
4h_1d_camarilla_breakout_v2
Uses Camarilla pivot levels from daily timeframe for breakout signals on 4h.
Long when price breaks above R4 level, short when breaks below S4 level.
Requires volume confirmation (>1.5x 20-period average) and ADX filter (>25 for trending).
Exit when price returns to pivot point (PP) or opposite S1/R1 level.
Designed for low trade frequency (target: 20-40 trades/year) to minimize fee drag.
Works in both trending and ranging markets by combining institutional pivot levels with volume confirmation.
"""

name = "4h_1d_camarilla_breakout_v2"
timeframe = "4h"
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
    
    # Get daily data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each daily bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla pivot calculation
    # Pivot Point (PP) = (High + Low + Close) / 3
    pp = (high_1d + low_1d + close_1d) / 3.0
    
    # Range = High - Low
    range_1d = high_1d - low_1d
    
    # Resistance levels
    r1 = close_1d + (range_1d * 1.1 / 12)
    r2 = close_1d + (range_1d * 1.1 / 6)
    r3 = close_1d + (range_1d * 1.1 / 4)
    r4 = close_1d + (range_1d * 1.1 / 2)
    
    # Support levels
    s1 = close_1d - (range_1d * 1.1 / 12)
    s2 = close_1d - (range_1d * 1.1 / 6)
    s3 = close_1d - (range_1d * 1.1 / 4)
    s4 = close_1d - (range_1d * 1.1 / 2)
    
    # Align Camarilla levels to 4h
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    
    # Volume confirmation on 4h: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    # ADX filter for trending markets (using 14-period)
    # Calculate True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Calculate Directional Movement
    dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smooth TR and DM
    tr_period = 14
    atr = pd.Series(tr).rolling(window=tr_period, min_periods=tr_period).mean().values
    dm_plus_smooth = pd.Series(dm_plus).rolling(window=tr_period, min_periods=tr_period).mean().values
    dm_minus_smooth = pd.Series(dm_minus).rolling(window=tr_period, min_periods=tr_period).mean().values
    
    # Calculate DI+ and DI-
    di_plus = np.where(atr != 0, (dm_plus_smooth / atr) * 100, 0)
    di_minus = np.where(atr != 0, (dm_minus_smooth / atr) * 100, 0)
    
    # Calculate DX and ADX
    dx = np.where((di_plus + di_minus) != 0, np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx = pd.Series(dx).rolling(window=tr_period, min_periods=tr_period).mean().values
    
    # ADX filter: >25 indicates trending market
    adx_filter = adx > 25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(pp_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(vol_confirm[i]) or 
            np.isnan(adx_filter[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: price breaks above R4 with volume and ADX filter
        if close[i] > r4_aligned[i] and vol_confirm[i] and adx_filter[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short entry: price breaks below S4 with volume and ADX filter
        elif close[i] < s4_aligned[i] and vol_confirm[i] and adx_filter[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions
        elif position == 1 and (close[i] <= pp_aligned[i] or close[i] <= s1_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] >= pp_aligned[i] or close[i] >= r1_aligned[i]):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals