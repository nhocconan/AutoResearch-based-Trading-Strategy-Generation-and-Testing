#!/usr/bin/env python3
"""
12h_Camarilla_Pivot_V3
Hypothesis: 12-hour strategy using daily Camarilla pivot levels with volume confirmation and weekly trend filter.
Long when price breaks above daily R2 with volume > 1.8x average and price > weekly EMA100.
Short when price breaks below daily S2 with volume > 1.8x average and price < weekly EMA100.
Exit when price crosses opposite daily level OR volume falls below 1.3x average.
Uses higher timeframe pivots for better signal quality in both bull and bear markets.
Target: 15-30 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Camarilla_Pivot_V3"
timeframe = "12h"
leverage = 1.0

def calculate_ema(close, period):
    """Calculate EMA with proper handling"""
    if len(close) < period:
        return np.full_like(close, np.nan, dtype=float)
    
    ema = np.full_like(close, np.nan, dtype=float)
    alpha = 2.0 / (period + 1)
    ema[period-1] = np.mean(close[:period])
    for i in range(period, len(close)):
        ema[i] = alpha * close[i] + (1 - alpha) * ema[i-1]
    return ema

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily and weekly data for context
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 50 or len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate daily Pivot (using previous daily bar's data)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    # Daily support/resistance levels (Camarilla S2/R2)
    S2_1d = pivot_1d - (range_1d * 1.1 / 6)  # daily S2
    R2_1d = pivot_1d + (range_1d * 1.1 / 6)  # daily R2
    
    # Calculate weekly EMA for trend filter
    ema_100_1w = calculate_ema(df_1w['close'].values, 100)
    
    # Align indicators to 12-hour timeframe
    S2_1d_aligned = align_htf_to_ltf(prices, df_1d, S2_1d)
    R2_1d_aligned = align_htf_to_ltf(prices, df_1d, R2_1d)
    ema_100_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_100_1w)
    
    # Volume confirmation: 15-period average (12h bars)
    vol_ma = np.full(n, np.nan)
    for i in range(15, n):
        vol_ma[i] = np.mean(volume[i-15:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(S2_1d_aligned[i]) or np.isnan(R2_1d_aligned[i]) or 
            np.isnan(ema_100_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        price = close[i]
        S2 = S2_1d_aligned[i]
        R2 = R2_1d_aligned[i]
        trend_up_1w = price > ema_100_1w_aligned[i]
        
        if position == 1:  # Long
            # Exit: price crosses below daily S2 or volume drops below 1.3x average
            if price < S2 or vol_ratio < 1.3:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price crosses above daily R2 or volume drops below 1.3x average
            if price > R2 or vol_ratio < 1.3:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above daily R2 with volume expansion and uptrend on weekly
            if price > R2 and vol_ratio > 1.8 and trend_up_1w:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below daily S2 with volume expansion and downtrend on weekly
            elif price < S2 and vol_ratio > 1.8 and not trend_up_1w:
                position = -1
                signals[i] = -0.25
    
    return signals