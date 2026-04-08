#!/usr/bin/env python3
"""
1d_1w_camarilla_pivot_v1
Hypothesis: Daily strategy using weekly Camarilla pivot with volume confirmation and trend filter.
Long when price breaks above weekly R1 with volume > 2x average and price > weekly EMA200.
Short when price breaks below weekly S1 with volume > 2x average and price < weekly EMA200.
Exit when price crosses opposite weekly level OR volume falls below 1.5x average.
Target: 10-25 trades/year to avoid overtrading. Weekly timeframe reduces noise and improves trend reliability.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_camarilla_pivot_v1"
timeframe = "1d"
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
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for context
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    # Calculate weekly Pivot (using previous week's data)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    range_1w = high_1w - low_1w
    # Weekly support/resistance levels (Standard Camarilla S1/R1)
    S1_1w = pivot_1w - (range_1w * 1.1 / 12)  # Weekly S1
    R1_1w = pivot_1w + (range_1w * 1.1 / 12)  # Weekly R1
    
    # Calculate weekly EMA for trend filter
    ema_200_1w = calculate_ema(close_1w, 200)
    
    # Align indicators to daily timeframe
    S1_1w_aligned = align_htf_to_ltf(prices, df_1w, S1_1w)
    R1_1w_aligned = align_htf_to_ltf(prices, df_1w, R1_1w)
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Volume confirmation: 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(S1_1w_aligned[i]) or np.isnan(R1_1w_aligned[i]) or 
            np.isnan(ema_200_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        price = close[i]
        S1 = S1_1w_aligned[i]
        R1 = R1_1w_aligned[i]
        trend_up_1w = price > ema_200_1w_aligned[i]
        
        if position == 1:  # Long
            # Exit: price crosses below weekly S1 or volume drops below 1.5x average
            if price < S1 or vol_ratio < 1.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price crosses above weekly R1 or volume drops below 1.5x average
            if price > R1 or vol_ratio < 1.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above weekly R1 with volume expansion and uptrend on weekly
            if price > R1 and vol_ratio > 2.5 and trend_up_1w:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below weekly S1 with volume expansion and downtrend on weekly
            elif price < S1 and vol_ratio > 2.5 and not trend_up_1w:
                position = -1
                signals[i] = -0.25
    
    return signals