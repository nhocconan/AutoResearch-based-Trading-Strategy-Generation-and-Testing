#!/usr/bin/env python3
"""
6h_1d_ADX_Trend_Filter_V1
Hypothesis: Use 1-day ADX to filter trend strength; only trade 6h EMA crossovers when daily trend is strong (ADX > 25). This avoids whipsaw in sideways markets and captures strong trends in both bull and bear markets. Uses 6h EMA(9)/EMA(21) crossover with daily ADX filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)"""
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), np.maximum(np.roll(low, 1) - low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    atr = np.zeros_like(tr)
    dm_plus_smooth = np.zeros_like(dm_plus)
    dm_minus_smooth = np.zeros_like(dm_minus)
    
    # Initial values
    atr[period-1] = np.mean(tr[:period])
    dm_plus_smooth[period-1] = np.mean(dm_plus[:period])
    dm_minus_smooth[period-1] = np.mean(dm_minus[:period])
    
    # Wilder's smoothing
    for i in range(period, len(tr)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (period-1) + dm_plus[i]) / period
        dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (period-1) + dm_minus[i]) / period
    
    # Directional Indicators
    plus_di = 100 * dm_plus_smooth / atr
    minus_di = 100 * dm_minus_smooth / atr
    
    # DX and ADX
    dx = np.zeros_like(close)
    dx[period:] = 100 * np.abs(plus_di[period:] - minus_di[period:]) / (plus_di[period:] + minus_di[period:])
    
    adx = np.zeros_like(close)
    adx[2*period-1] = np.mean(dx[period:2*period])
    for i in range(2*period, len(dx)):
        adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data once for ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate ADX on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Align ADX to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # 6h EMA crossover
    close_6h = prices['close'].values
    ema9 = pd.Series(close_6h).ewm(span=9, adjust=False).values
    ema21 = pd.Series(close_6h).ewm(span=21, adjust=False).values
    
    # EMA crossover signal: 1 = golden cross, -1 = death cross
    ema_cross = np.zeros(n)
    for i in range(1, n):
        if ema9[i] > ema21[i] and ema9[i-1] <= ema21[i-1]:
            ema_cross[i] = 1  # golden cross
        elif ema9[i] < ema21[i] and ema9[i-1] >= ema21[i-1]:
            ema_cross[i] = -1  # death cross
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if ADX not available
        if np.isnan(adx_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        adx = adx_1d_aligned[i]
        cross_signal = ema_cross[i]
        
        if position == 0:
            # Enter long: golden cross + strong trend (ADX > 25)
            if cross_signal == 1 and adx > 25:
                signals[i] = 0.25
                position = 1
            # Enter short: death cross + strong trend (ADX > 25)
            elif cross_signal == -1 and adx > 25:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: death cross or trend weakening (ADX < 20)
            if cross_signal == -1 or adx < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: golden cross or trend weakening (ADX < 20)
            if cross_signal == 1 or adx < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1d_ADX_Trend_Filter_V1"
timeframe = "6h"
leverage = 1.0