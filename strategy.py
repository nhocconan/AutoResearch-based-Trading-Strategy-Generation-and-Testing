#!/usr/bin/env python3
"""
6h_1d_TCI_Trend_v1
Hypothesis: Use Trend-Correction Index (TCI) from 1d to identify pullbacks in stronger trends (ADX>25) on 6h. 
Buy when TCI < -0.1 (oversold in uptrend) and ADX > 25, sell when TCI > 0.1 (overbought in downtrend) and ADX > 25.
This captures trend continuation after pullbacks, working in both bull/bear by following the dominant trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_TCI_Trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 6h ADX for trend strength filter
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 30:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # True Range
    tr1 = high_6h - low_6h
    tr2 = np.abs(high_6h - np.roll(close_6h, 1))
    tr3 = np.abs(low_6h - np.roll(close_6h, 1))
    tr_6h = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_6h[0] = tr1[0]
    
    # Directional Movement
    up_move = high_6h - np.roll(high_6h, 1)
    down_move = np.roll(low_6h, 1) - low_6h
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_14 = pd.Series(tr_6h).rolling(window=14, min_periods=14).mean().values
    plus_dm_14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    
    # DI and ADX
    plus_di = 100 * plus_dm_14 / tr_14
    minus_di = 100 * minus_dm_14 / tr_14
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_6h = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_6h_aligned = align_htf_to_ltf(prices, df_6h, adx_6h)
    
    # Daily Trend-Correction Index (TCI)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Linear regression slope (20-period) as trend component
    def linreg_slope(arr, window):
        slope = np.full_like(arr, np.nan, dtype=float)
        for i in range(window-1, len(arr)):
            y = arr[i-window+1:i+1]
            x = np.arange(window)
            if np.all(np.isnan(y)):
                slope[i] = np.nan
            else:
                # Use only valid points
                mask = ~np.isnan(y)
                if np.sum(mask) < 2:
                    slope[i] = np.nan
                else:
                    slope[i] = np.polyfit(x[mask], y[mask], 1)[0]
        return slope
    
    # 20-period linear regression slope of closes
    lr_slope = linreg_slope(close_1d, 20)
    
    # Standard deviation of residuals (detrended price)
    detrended = close_1d - np.roll(close_1d, 1)  # Simple change as detrended component
    std_dev = pd.Series(detrended).rolling(window=20, min_periods=20).std().values
    
    # TCI = (LR slope) / (std dev) - normalized trend strength
    tci = np.where(std_dev != 0, lr_slope / std_dev, 0)
    tci = np.nan_to_num(tci, nan=0.0)
    
    tci_aligned = align_htf_to_ltf(prices, df_1d, tci)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if any data invalid
        if (np.isnan(adx_6h_aligned[i]) or np.isnan(tci_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: only trade when ADX > 25 (trending market)
        trending = adx_6h_aligned[i] > 25
        
        # Trend-following on pullbacks
        long_signal = tci_aligned[i] < -0.1 and trending  # Oversold in uptrend
        short_signal = tci_aligned[i] > 0.1 and trending   # Overbought in downtrend
        
        # Exit: when TCI returns to neutral zone
        long_exit = tci_aligned[i] >= -0.05
        short_exit = tci_aligned[i] <= 0.05
        
        # Signal logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals