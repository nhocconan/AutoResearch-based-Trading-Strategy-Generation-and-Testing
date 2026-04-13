#!/usr/bin/env python3
"""
6h_1D_ADX_Trend_Filter_Cross_Signal_v1
Hypothesis: Use 1d ADX to filter trend strength, and 6h EMA crossover for entry. 
In trending markets (ADX > 25), trade EMA(9) x EMA(21) crossovers. 
In ranging markets (ADX <= 25), stay flat. 
This avoids whipsaws in low-volatility periods and captures strong trends.
Works in both bull and bear markets by only trading when trend is strong.
"""

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
    
    # Calculate ADX on 1d timeframe (trend strength filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # True Range
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    dm_plus = df_1d['high'].diff()
    dm_minus = df_1d['low'].diff().abs() * -1  # negative values
    dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
    dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)
    
    # Smoothed values
    atr = tr.rolling(window=14, min_periods=14).mean()
    dm_plus_smooth = dm_plus.rolling(window=14, min_periods=14).mean()
    dm_minus_smooth = dm_minus.rolling(window=14, min_periods=14).mean()
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    
    # ADX
    dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = dx.rolling(window=14, min_periods=14).mean()
    
    # Align ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx.values)
    
    # EMA crossover signals on 6h
    ema9 = pd.Series(close).ewm(span=9, min_periods=9, adjust=False).mean().values
    ema21 = pd.Series(close).ewm(span=21, min_periods=21, adjust=False).mean().values
    
    # Crossover detection
    ema_diff = ema9 - ema21
    ema_diff_prev = np.roll(ema_diff, 1)
    ema_diff_prev[0] = np.nan
    
    golden_cross = (ema_diff > 0) & (ema_diff_prev <= 0)
    death_cross = (ema_diff < 0) & (ema_diff_prev >= 0)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(30, n):  # warmup period
        # Skip if ADX not ready
        if np.isnan(adx_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Trend filter: only trade when ADX > 25 (strong trend)
        strong_trend = adx_aligned[i] > 25
        
        if not strong_trend:
            # No trend - stay flat
            signals[i] = 0.0
            position = 0
            continue
        
        # In strong trend, trade EMA crossovers
        if golden_cross[i]:
            # Golden cross - go long
            if position != 1:
                position = 1
                signals[i] = position_size
            else:
                signals[i] = position_size
        elif death_cross[i]:
            # Death cross - go short
            if position != -1:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = -position_size
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "6h_1D_ADX_Trend_Filter_Cross_Signal_v1"
timeframe = "6h"
leverage = 1.0