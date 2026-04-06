#!/usr/bin/env python3
"""
1h Supertrend trend filter with 4h/1d EMA confluence and volume confirmation.
Hypothesis: Supertrend captures trend direction, while 4h/1d EMA alignment and volume confirmation filter false signals, reducing whipsaws in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14294_1h_supertrend4h1d_vol_v1"
timeframe = "1h"
leverage = 1.0

def supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    
    hl2 = (high + low) / 2
    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr
    
    upper = np.where(np.isnan(upper), 0, upper)
    lower = np.where(np.isnan(lower), 0, lower)
    
    st = np.full_like(close, np.nan, dtype=float)
    dir = np.full_like(close, 1, dtype=int)
    
    for i in range(1, len(close)):
        if np.isnan(close[i-1]) or np.isnan(high[i]) or np.isnan(low[i]):
            st[i] = st[i-1]
            dir[i] = dir[i-1]
            continue
            
        if close[i] > upper[i-1]:
            dir[i] = 1
        elif close[i] < lower[i-1]:
            dir[i] = -1
        else:
            dir[i] = dir[i-1]
            if dir[i] == 1 and lower[i] < lower[i-1]:
                lower[i] = lower[i-1]
            if dir[i] == -1 and upper[i] > upper[i-1]:
                upper[i] = upper[i-1]
        
        st[i] = lower[i] if dir[i] == 1 else upper[i]
    
    return st, dir

def calculate_ema(close, period):
    """Calculate EMA with proper min_periods"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 4h data for Supertrend (once before loop)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Supertrend on 4h
    st_4h, dir_4h = supertrend(high_4h, low_4h, close_4h, 10, 3.0)
    
    # Load 1d data for EMA (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(50)
    ema_1d = calculate_ema(close_1d, 50)
    
    # Align to 1h timeframe
    st_4h_aligned = align_htf_to_ltf(prices, df_4h, st_4h)
    dir_4h_aligned = align_htf_to_ltf(prices, df_4h, dir_4h)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 1h data
    close = prices['close'].values
    volume = prices['volume'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ma)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from warmup period
    start = max(20, 50) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(st_4h_aligned[i]) or np.isnan(dir_4h_aligned[i]) or \
           np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: Supertrend reversal or price crosses EMA
        if position == 1:  # long position
            if dir_4h_aligned[i] == -1 or close[i] < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            if dir_4h_aligned[i] == 1 or close[i] > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries: Supertrend direction aligned with EMA and volume
            long_setup = (dir_4h_aligned[i] == 1) and (close[i] > ema_1d_aligned[i]) and vol_confirm[i] and session_filter[i]
            short_setup = (dir_4h_aligned[i] == -1) and (close[i] < ema_1d_aligned[i]) and vol_confirm[i] and session_filter[i]
            
            if long_setup:
                signals[i] = 0.20
                position = 1
            elif short_setup:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
    
    return signals