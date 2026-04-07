#!/usr/bin/env python3
"""
1h_breakout_momentum_v1
Hypothesis: Use 1d ADX for trend strength and 4h Donchian breakout for entry timing on 1h timeframe. 
Go long when price breaks above 4h Donchian high with 1d ADX > 25 (strong trend), short when breaks below 4h Donchian low with 1d ADX > 25. 
Use session filter (08-20 UTC) to avoid low liquidity periods. Position size fixed at 0.20 to manage risk.
Works in bull/bear via ADX trend filter and breakout mechanics. Low trade frequency expected due to strict ADX and breakout conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_breakout_momentum_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 4h data for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 4h Donchian channels (20-period)
    high_max_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_min_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d ADX (14-period)
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.inf], np.maximum(tr1, np.maximum(tr2, tr3))])
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    # Smoothed values
    atr = pd.Series(tr).ewm(span=14, adjust=False).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=14, adjust=False).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=14, adjust=False).mean().values
    # DI values
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(span=14, adjust=False).mean().values
    
    # Align indicators to 1h timeframe
    high_max_4h_1h = align_htf_to_ltf(prices, df_4h, high_max_4h)
    low_min_4h_1h = align_htf_to_ltf(prices, df_4h, low_min_4h)
    adx_1d_1h = align_htf_to_ltf(prices, df_1d, adx)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available or outside session
        if (np.isnan(high_max_4h_1h[i]) or np.isnan(low_min_4h_1h[i]) or 
            np.isnan(adx_1d_1h[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Strong trend filter
        strong_trend = adx_1d_1h[i] > 25
        
        if position == 1:  # Long position
            # Exit if trend weakens or price breaks below Donchian low
            if not strong_trend or close[i] < low_min_4h_1h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit if trend weakens or price breaks above Donchian high
            if not strong_trend or close[i] > high_max_4h_1h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Long entry: price breaks above 4h Donchian high with strong trend
            if close[i] > high_max_4h_1h[i] and strong_trend:
                position = 1
                signals[i] = 0.20
            # Short entry: price breaks below 4h Donchian low with strong trend
            elif close[i] < low_min_4h_1h[i] and strong_trend:
                position = -1
                signals[i] = -0.20
    
    return signals