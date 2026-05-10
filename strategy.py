#!/usr/bin/env python3
# 1h_4h_1d_ADX_SuperTrend_Filtered_Breakout
# Hypothesis: Combines ADX trend strength with SuperTrend direction on 4h for signal direction, using 1h for entry timing.
# ADX > 25 filters for trending markets, avoiding chop. SuperTrend provides dynamic support/resistance.
# Volume confirmation ensures breakout validity. Session filter (08-20 UTC) reduces noise.
# Designed to work in both bull (ADX up + SuperTrend up) and bear (ADX up + SuperTrend down) markets by following 4h trend.
# Target: 20-40 trades/year (~80-160 total over 4 years) to stay within optimal trade frequency for 1h.

name = "1h_4h_1d_ADX_SuperTrend_Filtered_Breakout"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h ADX (14-period) for trend strength
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range
    tr1 = high_4h[1:] - low_4h[1:]
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index 0
    
    # Directional Movement
    dm_plus = np.where((high_4h[1:] - high_4h[:-1]) > (low_4h[:-1] - low_4h[1:]), 
                       np.maximum(high_4h[1:] - high_4h[:-1], 0), 0)
    dm_minus = np.where((low_4h[:-1] - low_4h[1:]) > (high_4h[1:] - high_4h[:-1]), 
                        np.maximum(low_4h[:-1] - low_4h[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    tr_ma = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    dm_plus_ma = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False).mean().values
    dm_minus_ma = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False).mean().values
    
    # DI and DX
    di_plus = 100 * dm_plus_ma / tr_ma
    di_minus = 100 * dm_minus_ma / tr_ma
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_4h, adx)
    
    # 4h SuperTrend (ATR=10, multiplier=3.0)
    atr = pd.Series(tr).ewm(alpha=1/10, adjust=False).mean().values  # ATR(10)
    upper_band = (high_4h + low_4h) / 2 + 3.0 * atr
    lower_band = (high_4h + low_4h) / 2 - 3.0 * atr
    
    supertrend = np.full_like(close_4h, np.nan)
    direction = np.full_like(close_4h, 1)  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(close_4h)):
        if close_4h[i] <= upper_band[i-1]:
            upper_band[i] = upper_band[i]
        else:
            upper_band[i] = (high_4h[i] + low_4h[i]) / 2 + 3.0 * atr[i]
        
        if close_4h[i] >= lower_band[i-1]:
            lower_band[i] = lower_band[i]
        else:
            lower_band[i] = (high_4h[i] + low_4h[i]) / 2 - 3.0 * atr[i]
        
        if close_4h[i] > upper_band[i-1]:
            direction[i] = 1
        elif close_4h[i] < lower_band[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
        
        if direction[i] == 1:
            supertrend[i] = lower_band[i]
        else:
            supertrend[i] = upper_band[i]
    
    supertrend_aligned = align_htf_to_ltf(prices, df_4h, supertrend)
    direction_aligned = align_htf_to_ltf(prices, df_4h, direction)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    # Session filter: 08:00-20:00 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient warmup for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(adx_aligned[i]) or np.isnan(supertrend_aligned[i]) or 
            np.isnan(direction_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: ADX > 25 (trending), SuperTrend uptrend, price above SuperTrend, volume confirmation, session active
            if (adx_aligned[i] > 25 and 
                direction_aligned[i] == 1 and 
                close[i] > supertrend_aligned[i] and 
                volume_filter[i] and 
                session_filter[i]):
                signals[i] = 0.20
                position = 1
            # Short: ADX > 25 (trending), SuperTrend downtrend, price below SuperTrend, volume confirmation, session active
            elif (adx_aligned[i] > 25 and 
                  direction_aligned[i] == -1 and 
                  close[i] < supertrend_aligned[i] and 
                  volume_filter[i] and 
                  session_filter[i]):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit: ADX drops below 20 (losing trend) OR SuperTrend flips down
            if (adx_aligned[i] < 20 or 
                direction_aligned[i] == -1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit: ADX drops below 20 (losing trend) OR SuperTrend flips up
            if (adx_aligned[i] < 20 or 
                direction_aligned[i] == 1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals