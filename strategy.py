#!/usr/bin/env python3
"""
1h_4h_1d_Camarilla_Breakout_TrendFilter_v1
Hypothesis: Use 1d Camarilla pivot breakouts (H4/L4) for direction, 4h ADX for trend strength filter, and 1h for precise entry timing. Designed for low-frequency, high-quality trades in both bull and bear markets by requiring strong trends (ADX>25) and volume confirmation. Target: 15-30 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_Camarilla_Breakout_TrendFilter_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === DAILY CAMARILLA PIVOT CALCULATION ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    H4 = np.zeros(len(df_1d))
    L4 = np.zeros(len(df_1d))
    H3 = np.zeros(len(df_1d))
    L3 = np.zeros(len(df_1d))
    
    for i in range(len(df_1d)):
        range_ = high_1d[i] - low_1d[i]
        if range_ <= 0:
            H4[i] = L4[i] = H3[i] = L3[i] = close_1d[i]
        else:
            H4[i] = close_1d[i] + range_ * 1.1 / 2
            L4[i] = close_1d[i] - range_ * 1.1 / 2
            H3[i] = close_1d[i] + range_ * 1.1 / 4
            L3[i] = close_1d[i] - range_ * 1.1 / 4
    
    # Align Camarilla levels to 1h timeframe
    H4_1h = align_htf_to_ltf(prices, df_1d, H4)
    L4_1h = align_htf_to_ltf(prices, df_1d, L4)
    H3_1h = align_htf_to_ltf(prices, df_1d, H3)
    L3_1h = align_htf_to_ltf(prices, df_1d, L3)
    
    # === 4h TREND FILTER (ADX) ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate ADX for 4h trend strength
    plus_dm = np.zeros(len(df_4h))
    minus_dm = np.zeros(len(df_4h))
    tr = np.zeros(len(df_4h))
    
    for i in range(1, len(df_4h)):
        high_diff = high_4h[i] - high_4h[i-1]
        low_diff = low_4h[i-1] - low_4h[i]
        
        plus_dm[i] = high_diff if high_diff > low_diff and high_diff > 0 else 0
        minus_dm[i] = low_diff if low_diff > high_diff and low_diff > 0 else 0
        
        tr[i] = max(high_4h[i] - low_4h[i], abs(high_4h[i] - close_4h[i-1]), abs(low_4h[i] - close_4h[i-1]))
    
    # Smooth the values using Wilder's smoothing
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
    adx_4h = smooth_wilder(dx, period)
    
    # Align ADX to 1h timeframe
    adx_1h = align_htf_to_ltf(prices, df_4h, adx_4h)
    
    # === VOLUME CONFIRMATION (1h) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if not ready
        if (np.isnan(H4_1h[i]) or np.isnan(L4_1h[i]) or 
            np.isnan(adx_1h[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Breakout conditions with trend filter
        # Long: Price breaks above H4 with volume + strong trend (ADX > 25)
        long_breakout = (close[i] > H4_1h[i]) and (vol_ratio[i] > 1.5) and (adx_1h[i] > 25)
        
        # Short: Price breaks below L4 with volume + strong trend (ADX > 25)
        short_breakout = (close[i] < L4_1h[i]) and (vol_ratio[i] > 1.5) and (adx_1h[i] > 25)
        
        # Exit: Price returns to opposite H3/L3 level or trend weakens significantly
        exit_long = (position == 1) and ((close[i] < L3_1h[i]) or (adx_1h[i] < 20))
        exit_short = (position == -1) and ((close[i] > H3_1h[i]) or (adx_1h[i] < 20))
        
        # Execute trades
        if long_breakout and position != 1:
            position = 1
            signals[i] = 0.20
        elif short_breakout and position != -1:
            position = -1
            signals[i] = -0.20
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals