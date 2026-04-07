#!/usr/bin/env python3
"""
12h Camarilla Pivot + Weekly Trend + Volume Confirmation
Long when price touches L3 in bull market (price > weekly EMA200), short when touches H3.
Short when price touches H3 in bear market (price < weekly EMA200), long when touches L3.
Volume must exceed 20-period average for confirmation.
Designed for low trade frequency (target: 50-150 total trades over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1w_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly trend filter (EMA200)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    weekly_close = df_1w['close'].values
    weekly_ema = pd.Series(weekly_close).ewm(span=200, adjust=False, min_periods=200).mean().values
    weekly_ema_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema)
    
    # Daily Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    d_high = df_1d['high'].values
    d_low = df_1d['low'].values
    d_close = df_1d['close'].values
    
    pivot = (d_high + d_low + d_close * 2) / 4
    range_ = d_high - d_low
    
    H3 = d_close + range_ * 1.1 / 4
    L3 = d_close - range_ * 1.1 / 4
    
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):
        if np.isnan(weekly_ema_aligned[i]) or np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        bull_trend = close[i] > weekly_ema_aligned[i]
        
        if position == 1:  # Long
            if close[i] < L3_aligned[i] or not bull_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            if close[i] > H3_aligned[i] or bull_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if volume[i] <= vol_ma[i]:
                signals[i] = 0.0
                continue
            
            if bull_trend:
                if close[i] <= L3_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                elif close[i] >= H3_aligned[i]:
                    position = -1
                    signals[i] = -0.25
            else:
                if close[i] >= H3_aligned[i]:
                    position = -1
                    signals[i] = -0.25
                elif close[i] <= L3_aligned[i]:
                    position = 1
                    signals[i] = 0.25
    
    return signals