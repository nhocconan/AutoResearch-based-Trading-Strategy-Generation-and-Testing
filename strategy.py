#!/usr/bin/env python3
"""
12h Camarilla pivot with 1d trend filter and volume confirmation.
Uses daily EMA50 for trend direction and daily Camarilla levels for entry.
In uptrend (price > daily EMA50): long at L3, short at H3.
In downtrend (price < daily EMA50): short at H3, long at L3.
Volume must be above 20-period average for confirmation.
Target: 50-150 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1d_trend_volume_v2"
timeframe = "12h"
levereage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === DAILY TREND FILTER ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    daily_close = df_1d['close'].values
    daily_ema = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    daily_ema_aligned = align_htf_to_ltf(prices, df_1d, daily_ema)
    
    # === DAILY CAMARILLA PIVOTS ===
    d_high = df_1d['high'].values
    d_low = df_1d['low'].values
    d_close = df_1d['close'].values
    
    pivot = (d_high + d_low + d_close * 2) / 4
    range_ = d_high - d_low
    
    # Camarilla levels
    H3 = d_close + range_ * 1.1 / 4
    L3 = d_close - range_ * 1.1 / 4
    
    # Align to 12h timeframe (use previous day's levels)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    
    # === VOLUME CONFIRMATION ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        if np.isnan(daily_ema_aligned[i]) or np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        # Determine trend from daily EMA
        uptrend = close[i] > daily_ema_aligned[i]
        
        if position == 1:  # Long
            # Exit: price crosses below L3 OR trend turns down
            if close[i] < L3_aligned[i] or not uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price crosses above H3 OR trend turns up
            if close[i] > H3_aligned[i] or uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Need volume confirmation
            if volume[i] <= vol_ma[i]:
                signals[i] = 0.0
                continue
            
            # Entry based on trend
            if uptrend:
                # Uptrend: long at L3 support, short at H3 resistance
                if close[i] <= L3_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                elif close[i] >= H3_aligned[i]:
                    position = -1
                    signals[i] = -0.25
            else:
                # Downtrend: short at H3 resistance, long at L3 support
                if close[i] >= H3_aligned[i]:
                    position = -1
                    signals[i] = -0.25
                elif close[i] <= L3_aligned[i]:
                    position = 1
                    signals[i] = 0.25
    
    return signals

leverage = 1.0