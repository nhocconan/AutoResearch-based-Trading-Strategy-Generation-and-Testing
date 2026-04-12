#!/usr/bin/env python3
"""
4h_1d_camarilla_breakout_v26
Uses 1-day Camarilla pivot levels on the 4h timeframe with volume and ADX trend filter.
Long: price breaks above H3 with ADX > 25 and volume > 1.5x average.
Short: price breaks below L3 with ADX > 25 and volume > 1.5x average.
Exit: price returns to Pivot point or ADX drops below 20.
Designed for 4h timeframe with target 25-40 trades/year to minimize fee drag.
Works in trending markets (ADX > 25) by following institutional pivot levels.
"""

name = "4h_1d_camarilla_breakout_v26"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index"""
    plus_dm = np.zeros_like(high)
    minus_dm = np.zeros_like(high)
    tr = np.zeros_like(high)
    
    for i in range(1, len(high)):
        plus_dm[i] = max(high[i] - high[i-1], 0)
        minus_dm[i] = max(low[i-1] - low[i], 0)
        if plus_dm[i] < minus_dm[i]:
            plus_dm[i] = 0
        if minus_dm[i] < plus_dm[i]:
            minus_dm[i] = 0
        
        tr[i] = max(high[i] - low[i], 
                   abs(high[i] - close[i-1]), 
                   abs(low[i] - close[i-1]))
    
    # Wilder's smoothing
    atr = np.zeros_like(high)
    atr[period] = np.mean(tr[1:period+1])
    for i in range(period+1, len(high)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    
    plus_di = np.zeros_like(high)
    minus_di = np.zeros_like(high)
    dx = np.zeros_like(high)
    
    for i in range(period, len(high)):
        if atr[i] > 0:
            plus_di[i] = 100 * (plus_dm[i] / atr[i])
            minus_di[i] = 100 * (minus_dm[i] / atr[i])
            if plus_di[i] + minus_di[i] > 0:
                dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
    
    adx = np.zeros_like(high)
    adx[2*period] = np.mean(dx[period:2*period+1])
    for i in range(2*period+1, len(high)):
        adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    
    return adx

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels"""
    range_val = high - low
    close_last = close[-1]
    
    pivot = (high + low + close_last) / 3
    r4 = close_last + range_val * 1.1 / 2
    r3 = close_last + range_val * 1.1 / 4
    r2 = close_last + range_val * 1.1 / 6
    r1 = close_last + range_val * 1.1 / 12
    s1 = close_last - range_val * 1.1 / 12
    s2 = close_last - range_val * 1.1 / 6
    s3 = close_last - range_val * 1.1 / 4
    s4 = close_last - range_val * 1.1 / 2
    
    return {
        'P': pivot,
        'R4': r4, 'R3': r3, 'R2': r2, 'R1': r1,
        'S1': s1, 'S2': s2, 'S3': s3, 'S4': s4
    }

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
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX on 4h for trend strength
    adx = calculate_adx(high, low, close, 14)
    
    # Calculate Camarilla levels for each day
    camarilla_levels = {}
    for i in range(len(df_1d)):
        day_high = high_1d[i]
        day_low = low_1d[i]
        day_close = close_1d[i]
        camarilla_levels[i] = calculate_camarilla(day_high, day_low, day_close)
    
    # Prepare arrays for each level (same length as daily data)
    P = np.array([camarilla_levels[i]['P'] for i in range(len(df_1d))])
    R3 = np.array([camarilla_levels[i]['R3'] for i in range(len(df_1d))])
    L3 = np.array([camarilla_levels[i]['S3'] for i in range(len(df_1d))])
    R4 = np.array([camarilla_levels[i]['R4'] for i in range(len(df_1d))])
    L4 = np.array([camarilla_levels[i]['S4'] for i in range(len(df_1d))])
    
    # Align to 4h timeframe
    P_aligned = align_htf_to_ltf(prices, df_1d, P)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(P_aligned[i]) or np.isnan(R3_aligned[i]) or np.isnan(L3_aligned[i]) or
            np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: price breaks above R3 with ADX > 25 and volume confirmation
        if (close[i] > R3_aligned[i] and adx[i] > 25 and vol_confirm[i] and position != 1):
            position = 1
            signals[i] = 0.25
        # Short entry: price breaks below L3 with ADX > 25 and volume confirmation
        elif (close[i] < L3_aligned[i] and adx[i] > 25 and vol_confirm[i] and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit conditions: price returns to Pivot or ADX drops below 20
        elif position == 1 and (close[i] < P_aligned[i] or adx[i] < 20):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > P_aligned[i] or adx[i] < 20):
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