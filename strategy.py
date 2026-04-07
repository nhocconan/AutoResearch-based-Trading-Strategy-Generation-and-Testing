#!/usr/bin/env python3
"""
6h Keltner Breakout with 12h ADX Filter - Version 2
Hypothesis: Keltner channel breakouts capture volatility expansion, while 12h ADX > 25 filters for trending conditions, reducing whipsaws in ranging markets. This combination works in both bull and bear markets by only trading when volatility expands in the direction of the trend. Volume confirmation (>1.5x average) ensures institutional participation. Target: 25-35 trades/year (~100-140 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_keltner_breakout_12h_adx_filter_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Keltner Channel (20, 2.0)
    ma = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    atr = pd.Series(high - low).rolling(window=20, min_periods=20).mean().values
    upper = ma + (atr * 2.0)
    lower = ma - (atr * 2.0)
    
    # Volume filter (>1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    # 12h ADX (14-period) for trend strength
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((high_12h - np.roll(high_12h, 1)) > (np.roll(low_12h, 1) - low_12h), 
                       np.maximum(high_12h - np.roll(high_12h, 1), 0), 0)
    dm_minus = np.where((np.roll(low_12h, 1) - low_12h) > (high_12h - np.roll(high_12h, 1)), 
                        np.maximum(np.roll(low_12h, 1) - low_12h, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    dm_plus14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    # DI and DX
    di_plus = np.where(tr14 > 0, 100 * dm_plus14 / tr14, 0)
    di_minus = np.where(tr14 > 0, 100 * dm_minus14 / tr14, 0)
    dx = np.where((di_plus + di_minus) > 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    
    # ADX
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(adx_12h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below middle line OR ADX drops below 20
            if close[i] < ma[i] or adx_12h_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above middle line OR ADX drops below 20
            if close[i] > ma[i] or adx_12h_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: breakout above upper Keltner + ADX > 25 + volume filter
            if (close[i] > upper[i-1] and 
                adx_12h_aligned[i] > 25 and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short: breakout below lower Keltner + ADX > 25 + volume filter
            elif (close[i] < lower[i-1] and 
                  adx_12h_aligned[i] > 25 and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals