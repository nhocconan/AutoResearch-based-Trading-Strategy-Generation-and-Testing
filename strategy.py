#!/usr/bin/env python3
"""
1d Donchian Breakout with 1w Trend and Volume Confirmation.
Long when price breaks above upper Donchian(20) with 1w uptrend and volume confirmation.
Short when price breaks below lower Donchian(20) with 1w downtrend and volume confirmation.
Exit when price crosses back below midpoint (long) or above midpoint (short).
Target: 30-100 total trades over 4 years (7-25/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian_breakout_1w_trend_volume_v1"
timeframe = "1d"
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
    
    # === 1W EMA TREND FILTER (HTF) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    one_w_close = df_1w['close'].values
    one_w_ema = pd.Series(one_w_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    one_w_ema_aligned = align_htf_to_ltf(prices, df_1w, one_w_ema)
    
    # === DAILY DONCHIAN CHANNEL (20) ===
    # Using previous 20 days (excluding current)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    midpoint = np.full(n, np.nan)
    
    for i in range(20, n):
        upper[i] = np.max(high[i-20:i])
        lower[i] = np.min(low[i-20:i])
        midpoint[i] = (upper[i] + lower[i]) / 2
    
    # === VOLUME CONFIRMATION (DAILY) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(midpoint[i]) or 
            np.isnan(one_w_ema_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from 1w EMA
        uptrend = close[i] > one_w_ema_aligned[i]
        downtrend = close[i] < one_w_ema_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below midpoint OR trend turns down
            if close[i] < midpoint[i] or downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above midpoint OR trend turns up
            if close[i] > midpoint[i] or uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need volume confirmation
            if volume[i] <= vol_ma[i]:
                signals[i] = 0.0
                continue
            
            # Entry: Donchian breakout with trend alignment
            if close[i] > upper[i] and uptrend:
                # Breakout above upper Donchian in uptrend -> long
                position = 1
                signals[i] = 0.25
            elif close[i] < lower[i] and downtrend:
                # Breakdown below lower Donchian in downtrend -> short
                position = -1
                signals[i] = -0.25
    
    return signals