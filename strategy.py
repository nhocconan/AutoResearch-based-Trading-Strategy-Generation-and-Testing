#!/usr/bin/env python3
"""
12h_1w_KAMA_Trend_Follow
Hypothesis: 12-hour KAMA (Kaufman Adaptive Moving Average) trend with weekly ADX filter and volume confirmation.
KAMA adapts to market noise, reducing whipsaws in sideways markets. Weekly ADX > 25 ensures strong trend context.
Volume > 1.5x 20-period average confirms breakout strength. Designed for low-frequency trades (12-37/year) to avoid fee drag.
Works in bull markets by catching trends and in bear markets by avoiding false signals during consolidation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_KAMA_Trend_Follow"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === WEEKLY ADX TREND FILTER ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ADX on weekly data
    plus_dm = np.zeros(len(df_1w))
    minus_dm = np.zeros(len(df_1w))
    tr = np.zeros(len(df_1w))
    
    for i in range(1, len(df_1w)):
        high_diff = high_1w[i] - high_1w[i-1]
        low_diff = low_1w[i-1] - low_1w[i]
        
        plus_dm[i] = high_diff if high_diff > low_diff and high_diff > 0 else 0
        minus_dm[i] = low_diff if low_diff > high_diff and low_diff > 0 else 0
        
        tr[i] = max(high_1w[i] - low_1w[i], abs(high_1w[i] - close_1w[i-1]), abs(low_1w[i] - close_1w[i-1]))
    
    # Wilder smoothing
    def smooth_wilder(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        result[period-1] = np.nansum(arr[1:period+1])
        for i in range(period, len(arr)):
            result[i] = result[i-1] - (result[i-1] / period) + arr[i]
        return result
    
    period = 14
    tr_smooth = smooth_wilder(tr, period)
    plus_di = 100 * smooth_wilder(plus_dm, period) / tr_smooth
    minus_di = 100 * smooth_wilder(minus_dm, period) / tr_smooth
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_1w = smooth_wilder(dx, period)
    
    # Align weekly ADX to 12h timeframe
    adx_12h = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # === KAMA ON 12H DATA ===
    # Efficiency Ratio (ER)
    change = np.abs(np.diff(close, k=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # 10-period volatility
    # Handle first 10 values
    change = np.concatenate([np.full(10, np.nan), change])
    volatility = np.concatenate([np.full(10, np.nan), volatility])
    
    er = np.where(volatility != 0, change / volatility, 0)
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # Start after 10 periods
    for i in range(10, n):
        if np.isnan(kama[i-1]) or np.isnan(sc[i]):
            kama[i] = close[i]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === VOLUME CONFIRMATION ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if not ready
        if (np.isnan(kama[i]) or np.isnan(adx_12h[i]) or 
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Entry conditions
        # Long: Price above KAMA + strong weekly trend + volume confirmation
        long_entry = (close[i] > kama[i]) and (adx_12h[i] > 25) and (vol_ratio[i] > 1.5)
        
        # Short: Price below KAMA + strong weekly trend + volume confirmation
        short_entry = (close[i] < kama[i]) and (adx_12h[i] > 25) and (vol_ratio[i] > 1.5)
        
        # Exit: Opposite KAMA cross or trend weakening
        exit_long = (position == 1) and (close[i] < kama[i])
        exit_short = (position == -1) and (close[i] > kama[i])
        
        # Execute trades
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals