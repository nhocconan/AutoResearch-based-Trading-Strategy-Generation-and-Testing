#!/usr/bin/env python3
"""
4h_Camarilla_Pivot_V1
Hypothesis: Camarilla pivot levels from 1-day timeframe act as significant support/resistance.
Breakouts above R1 or below S1 with volume confirmation and ADX trend filter provide high-probability trades.
Works in both bull and bear markets by only taking breakout trades aligned with daily trend (price above/below 200 EMA).
Designed for low trade frequency (~25-35 trades/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load 1d data once for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Camarilla pivots from previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    R4 = close_1d + (high_1d - low_1d) * 1.500
    R3 = close_1d + (high_1d - low_1d) * 1.250
    R2 = close_1d + (high_1d - low_1d) * 1.166
    R1 = close_1d + (high_1d - low_1d) * 1.083
    S1 = close_1d - (high_1d - low_1d) * 1.083
    S2 = close_1d - (high_1d - low_1d) * 1.166
    S3 = close_1d - (high_1d - low_1d) * 1.250
    S4 = close_1d - (high_1d - low_1d) * 1.500
    
    # Align Camarilla levels to 4h timeframe
    R1_4h = align_htf_to_ltf(prices, df_1d, R1)
    S1_4h = align_htf_to_ltf(prices, df_1d, S1)
    R2_4h = align_htf_to_ltf(prices, df_1d, R2)
    S2_4h = align_htf_to_ltf(prices, df_1d, S2)
    
    # Calculate 1d 200 EMA for trend filter
    close_1d_series = pd.Series(close_1d)
    ema200_1d = close_1d_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_4h = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Calculate ADX for trend strength (using 1d data)
    high_1d_series = pd.Series(high_1d)
    low_1d_series = pd.Series(low_1d)
    close_1d_series = pd.Series(close_1d)
    
    # True Range
    tr1 = high_1d_series - low_1d_series
    tr2 = abs(high_1d_series - close_1d_series.shift(1))
    tr3 = abs(low_1d_series - close_1d_series.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    up_move = high_1d_series - high_1d_series.shift(1)
    down_move = low_1d_series.shift(1) - low_1d_series
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_dm_14 = pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    minus_dm_14 = pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # DI and DX
    plus_di = 100 * plus_dm_14 / tr_14
    minus_di = 100 * minus_dm_14 / tr_14
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_4h = align_htf_to_ltf(prices, df_1d, adx)
    
    # 4h price data
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    volume_4h = prices['volume'].values
    
    # Volume filter: 20-period average
    vol_ma_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Skip if any critical values are NaN
        if (np.isnan(R1_4h[i]) or np.isnan(S1_4h[i]) or np.isnan(ema200_4h[i]) or 
            np.isnan(adx_4h[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_4h[i]
        vol = volume_4h[i]
        r1 = R1_4h[i]
        s1 = S1_4h[i]
        ema200 = ema200_4h[i]
        adx_val = adx_4h[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation: current volume above average
        vol_ok = vol > vol_ma
        
        # Trend filter: ADX > 25 indicates trending market
        trend_ok = adx_val > 25
        
        if position == 0:
            # Long breakout: price breaks above R1 with volume, trend up (price > EMA200)
            if price > r1 and vol_ok and trend_ok and price > ema200:
                signals[i] = 0.25
                position = 1
            # Short breakout: price breaks below S1 with volume, trend down (price < EMA200)
            elif price < s1 and vol_ok and trend_ok and price < ema200:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below S1 or ADX weakens
            if price < s1 or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above R1 or ADX weakens
            if price > r1 or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_Pivot_V1"
timeframe = "4h"
leverage = 1.0