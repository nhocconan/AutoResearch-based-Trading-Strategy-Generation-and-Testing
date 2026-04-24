#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3/S3 breakout with 1d volume spike and 1d ADX regime filter.
- Primary timeframe: 4h for execution, HTF: 1d for volume confirmation and ADX trend strength.
- Camarilla levels: calculated from prior 1d OHLC (R3, S3, R4, S4).
- Breakout logic: Long when close > R3 and volume spike, Short when close < S3 and volume spike.
- ADX filter: Only take breakout trades when 1d ADX > 25 (strong trend), avoid ranging markets (ADX < 20).
- Volume confirmation: 1d volume > 1.8 * 20-period 1d volume MA (to ensure institutional participation).
- Discrete signal size: 0.25 to balance profit potential and drawdown control.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for Camarilla levels, volume, and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Camarilla levels from prior 1d OHLC
    # R4 = Close + 1.1 * (High - Low) * 1.1/2
    # R3 = Close + 1.1 * (High - Low) * 1.1/4
    # S3 = Close - 1.1 * (High - Low) * 1.1/4
    # S4 = Close - 1.1 * (High - Low) * 1.1/2
    hl_range = df_1d['high'] - df_1d['low']
    close_1d = df_1d['close']
    r3 = close_1d + 1.1 * hl_range * (1.1/4)
    s3 = close_1d - 1.1 * hl_range * (1.1/4)
    r4 = close_1d + 1.1 * hl_range * (1.1/2)
    s4 = close_1d - 1.1 * hl_range * (1.1/2)
    
    # Align Camarilla levels to 4h (use prior day's levels for today's trading)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3.values)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3.values)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4.values)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4.values)
    
    # Calculate 1d ADX (14-period)
    # True Range
    tr1 = pd.Series(df_1d['high']).diff().abs()
    tr2 = (pd.Series(df_1d['high']) - pd.Series(df_1d['low'].shift())).abs()
    tr3 = (pd.Series(df_1d['low']) - pd.Series(df_1d['close'].shift())).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Movement
    up_move = pd.Series(df_1d['high']).diff()
    down_move = -pd.Series(df_1d['low']).diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / (atr + 1e-10)
    minus_di = 100 * minus_dm_smooth / (atr + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align ADX to 4h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: 1d volume > 1.8 * 20-period volume MA
    volume_ma = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    volume_spike = df_1d['volume'].values > (1.8 * volume_ma)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 30  # Need enough 1d bars for ADX/volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_spike_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        r4_val = r4_aligned[i]
        s4_val = s4_aligned[i]
        adx_val = adx_aligned[i]
        vol_spike = volume_spike_aligned[i]
        
        if position == 0:
            # Check for entry signals
            if vol_spike and adx_val > 25:  # Strong trend + volume confirmation
                # Breakout entries
                if price > r3_val:
                    signals[i] = 0.25
                    position = 1
                elif price < s3_val:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price reaches R4 (take profit) or reverses below R3 (stop loss)
            if price >= r4_val or price < r3_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reaches S4 (take profit) or reverses above S3 (stop loss)
            if price <= s4_val or price > s3_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1dADX_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0