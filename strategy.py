#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Load 1w data once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate daily pivot points (R2, S2)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = 2 * pivot - low_1d
    s1 = 2 * pivot - high_1d
    r2 = pivot + (high_1d - low_1d)
    s2 = pivot - (high_1d - low_1d)
    
    # Calculate 1-week EMA for trend filter
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Volume filter: 30-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=30, min_periods=30).mean().values
    
    # Pre-calculate constant values for alignment (using previous day's data)
    r2_prev = np.append([np.nan], r2[:-1])
    s2_prev = np.append([np.nan], s2[:-1])
    pivot_prev = np.append([np.nan], pivot[:-1])
    ema_prev = np.append([np.nan], ema_1w[:-1])
    
    # Align to 12h timeframe
    r2_12h = align_htf_to_ltf(prices, df_1d, r2_prev)
    s2_12h = align_htf_to_ltf(prices, df_1d, s2_prev)
    pivot_12h = align_htf_to_ltf(prices, df_1d, pivot_prev)
    ema_12h = align_htf_to_ltf(prices, df_1w, ema_prev)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(100, n):
        # Skip if any critical data is NaN
        if np.isnan(vol_ma[i]) or np.isnan(r2_12h[i]) or np.isnan(s2_12h[i]) or np.isnan(pivot_12h[i]) or np.isnan(ema_12h[i]):
            continue
        
        if position == 0:
            # Long: price breaks above R2 + price above weekly EMA (bullish trend) + volume confirmation
            if (close[i] > r2_12h[i] and  # price breaks above R2 resistance
                close[i] > ema_12h[i] and  # price above weekly EMA (bullish trend)
                volume[i] > vol_ma[i] * 1.5):  # volume confirmation
                position = 1
                signals[i] = position_size
            # Short: price breaks below S2 + price below weekly EMA (bearish trend) + volume confirmation
            elif (close[i] < s2_12h[i] and  # price breaks below S2 support
                  close[i] < ema_12h[i] and  # price below weekly EMA (bearish trend)
                  volume[i] > vol_ma[i] * 1.5):  # volume confirmation
                position = -1
                signals[i] = -position_size
        elif position == 1:
            # Exit: price crosses below pivot or weekly EMA
            if close[i] < pivot_12h[i] or close[i] < ema_12h[i]:
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Exit: price crosses above pivot or weekly EMA
            if close[i] > pivot_12h[i] or close[i] > ema_12h[i]:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "12h_Custom_Pivot_Trend_Filter_v2"
timeframe = "12h"
leverage = 1.0