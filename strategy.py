#!/usr/bin/env python3
"""
4h_TrendFilter_RangeReversion
Hypothesis: In 2025-2026 bear/range market, price often reverses from extreme ranges.
Strategy: Use 4h Bollinger Bands (20,2) for overbought/oversold signals.
Filter trades with 1d ADX < 20 (range market) and volume > 1.5x 20 EMA.
Exit when price returns to BB middle (20 SMA).
Designed for low trade frequency (<30/year) and works in ranging markets.
"""

name = "4h_TrendFilter_RangeReversion"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for ADX (range filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d ADX (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align to same length
    
    # Plus Directional Movement (+DM) and Minus Directional Movement (-DM)
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed values
    tr14 = pd.Series(tr).ewm(span=14, adjust=False).mean().values
    plus_dm14 = pd.Series(plus_dm).ewm(span=14, adjust=False).mean().values
    minus_dm14 = pd.Series(minus_dm).ewm(span=14, adjust=False).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm14 / tr14
    minus_di = 100 * minus_dm14 / tr14
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(span=14, adjust=False).mean().values
    
    # Align ADX to 4h timeframe (need trend from previous day)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # 4h Bollinger Bands (20,2)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    sma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = sma20 + 2 * std20
    lower_bb = sma20 - 2 * std20
    
    # Volume filter: current volume > 1.5x 20 EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ema20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 1d ADX (20), 4h BB (20), vol EMA (20)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(sma20[i]) or
            np.isnan(upper_bb[i]) or
            np.isnan(lower_bb[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Range filter: only trade when ADX < 20 (no strong trend)
        range_market = adx_aligned[i] < 20
        
        if position == 0 and range_market:
            # Long: price at or below lower BB with volume confirmation
            if low[i] <= lower_bb[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price at or above upper BB with volume confirmation
            elif high[i] >= upper_bb[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price returns to middle (SMA20)
            if close[i] >= sma20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to middle (SMA20)
            if close[i] <= sma20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals