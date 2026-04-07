#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla pivot level touch + volume confirmation + weekly trend filter.
Uses weekly EMA200 for trend direction (bull/bear filter) and daily Camarilla levels for entry.
In bull markets (price > weekly EMA200): long at L3, short at H3.
In bear markets (price < weekly EMA200): short at H3, long at L3.
Volume must be above 20-period average to confirm breakout.
Low trade frequency expected due to specific pivot level touches.
Works in both bull/bear by adapting direction based on higher timeframe trend.
Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1w_trend_volume_v1"
timeframe = "12h"
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
    
    # === WEEKLY TREND FILTER (HTF) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    weekly_close = df_1w['close'].values
    weekly_ema = pd.Series(weekly_close).ewm(span=200, adjust=False, min_periods=200).mean().values
    weekly_ema_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema)  # already shifted
    
    # === DAILY CAMARILLA PIVOTS (HTF) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    d_high = df_1d['high'].values
    d_low = df_1d['low'].values
    d_close = df_1d['close'].values
    
    pivot = (d_high + d_low + d_close * 2) / 4
    range_ = d_high - d_low
    
    # Camarilla levels
    H4 = d_close + range_ * 1.1 / 2
    H3 = d_close + range_ * 1.1 / 4
    L3 = d_close - range_ * 1.1 / 4
    L4 = d_close - range_ * 1.1 / 2
    
    # Align to 12h timeframe (use previous day's levels)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    
    # === VOLUME CONFIRMATION (LTF) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):  # Start after warmup
        if np.isnan(weekly_ema_aligned[i]) or np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from weekly EMA
        bull_trend = close[i] > weekly_ema_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below L3 OR weekly trend turns bearish
            if close[i] < L3_aligned[i] or not bull_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above H3 OR weekly trend turns bullish
            if close[i] > H3_aligned[i] or bull_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need volume confirmation
            if volume[i] <= vol_ma[i]:
                signals[i] = 0.0
                continue
            
            # Entry logic based on weekly trend
            if bull_trend:
                # In bull market: long at L3 support, short at H3 resistance
                if close[i] <= L3_aligned[i]:  # Touch or break below L3 -> long
                    position = 1
                    signals[i] = 0.25
                elif close[i] >= H3_aligned[i]:  # Touch or break above H3 -> short
                    position = -1
                    signals[i] = -0.25
            else:
                # In bear market: short at H3 resistance, long at L3 support
                if close[i] >= H3_aligned[i]:  # Touch or break above H3 -> short
                    position = -1
                    signals[i] = -0.25
                elif close[i] <= L3_aligned[i]:  # Touch or break below L3 -> long
                    position = 1
                    signals[i] = 0.25
    
    return signals