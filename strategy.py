#!/usr/bin/env python3
"""
4h_1d_camarilla_breakout_v2
Hypothesis: Uses Camarilla pivot levels from 1d to identify key support/resistance levels.
Enters on break of L3 (long) or H3 (short) with volume confirmation.
Uses ADX to filter for trending markets only (ADX > 25).
Works in both bull and bear markets by trading breakouts from key levels.
Target: 20-50 trades/year on 4h (80-200 total over 4 years).
"""

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
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    # Formula: 
    # H4 = Close + 1.5 * (High - Low)
    # H3 = Close + 1.0 * (High - Low)
    # H2 = Close + 0.5 * (High - Low)
    # H1 = Close + 0.25 * (High - Low)
    # L1 = Close - 0.25 * (High - Low)
    # L2 = Close - 0.5 * (High - Low)
    # L3 = Close - 1.0 * (High - Low)
    # L4 = Close - 1.5 * (High - Low)
    
    # Use previous day's data to avoid look-ahead
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate Camarilla levels
    H3 = prev_close + 1.0 * (prev_high - prev_low)
    L3 = prev_close - 1.0 * (prev_high - prev_low)
    
    # Get 4h data for price and volume
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate ADX for trend filtering (14-period)
    # +DM and -DM
    up_move = high_4h[1:] - high_4h[:-1]
    down_move = low_4h[:-1] - low_4h[1:]
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # True Range
    tr1 = high_4h[1:] - low_4h[1:]
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.mean(data[:period])
            for i in range(period, len(data)):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    period = 14
    if len(tr) >= period:
        atr = wilders_smoothing(tr, period)
        plus_di = 100 * wilders_smoothing(plus_dm, period) / atr
        minus_di = 100 * wilders_smoothing(minus_dm, period) / atr
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = wilders_smoothing(dx, period)
    else:
        adx = np.full(len(tr), np.nan)
    
    # Align all to 4h timeframe (adx needs special handling due to calculation shift)
    # Camarilla levels from previous day - already aligned to 4h bars
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    
    # Volume moving average
    vol_ma_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% of capital
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(adx[i-1] if i-1 < len(adx) else np.nan)):
            signals[i] = 0.0
            continue
        
        # Trend filter: ADX > 25
        trending = adx[i-1] > 25 if i-1 < len(adx) else False
        
        # Volume confirmation
        volume_expansion = volume_4h[i] > (vol_ma_20[i] * 1.5)
        
        # Breakout conditions
        long_breakout = close_4h[i] > H3_aligned[i] and volume_expansion and trending
        short_breakout = close_4h[i] < L3_aligned[i] and volume_expansion and trending
        
        # Exit conditions: reverse signal or loss of momentum
        long_exit = close_4h[i] < L3_aligned[i]  # Reverse to short level
        short_exit = close_4h[i] > H3_aligned[i]  # Reverse to long level
        
        if long_breakout and position != 1:
            position = 1
            signals[i] = position_size
        elif short_breakout and position != -1:
            position = -1
            signals[i] = -position_size
        elif long_exit and position == 1:
            position = 0
            signals[i] = 0.0
        elif short_exit and position == -1:
            position = 0
            signals[i] = 0.0
        elif position == 1:
            signals[i] = position_size
        elif position == -1:
            signals[i] = -position_size
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_1d_camarilla_breakout_v2"
timeframe = "4h"
leverage = 1.0