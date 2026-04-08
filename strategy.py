#!/usr/bin/env python3
"""
1d_1w_camarilla_pivot_v8
Hypothesis: Daily strategy using weekly context with Camarilla pivot levels.
Long when price crosses above weekly Pivot with volume > 2.0x average and price > weekly EMA200 (bullish trend).
Short when price crosses below weekly Pivot with volume > 2.0x average and price < weekly EMA200 (bearish trend).
Exit when price crosses opposite weekly support/resistance or volume drops below average.
Uses discrete position sizing (0.30) to balance return and risk. Target: 10-20 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_camarilla_pivot_v8"
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

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels"""
    if len(high) < 1:
        return np.full(len(high), np.nan), np.full(len(high), np.nan), \
               np.full(len(high), np.nan), np.full(len(high), np.nan)
    
    pivot = (high + low + close) / 3.0
    range_val = high - low
    
    # Standard Camarilla multipliers
    S1 = pivot - (range_val * 1.1 / 12)
    R1 = pivot + (range_val * 1.1 / 12)
    S2 = pivot - (range_val * 1.1 / 6)
    R2 = pivot + (range_val * 1.1 / 6)
    
    return pivot, S1, R1, S2, R2

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for context and Pivot
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    # Calculate weekly Pivot (using previous week's data)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot_1w, S1_1w, R1_1w, S2_1w, R2_1w = calculate_camarilla(high_1w, low_1w, close_1w)
    
    # Calculate weekly EMA for trend filter
    ema_200_1w = calculate_ema(close_1w, 200)
    
    # Align indicators to daily timeframe
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    S1_1w_aligned = align_htf_to_ltf(prices, df_1w, S1_1w)
    R1_1w_aligned = align_htf_to_ltf(prices, df_1w, R1_1w)
    S2_1w_aligned = align_htf_to_ltf(prices, df_1w, S2_1w)
    R2_1w_aligned = align_htf_to_ltf(prices, df_1w, R2_1w)
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Volume confirmation: 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(pivot_1w_aligned[i]) or np.isnan(S1_1w_aligned[i]) or 
            np.isnan(R1_1w_aligned[i]) or np.isnan(ema_200_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        price = close[i]
        pivot = pivot_1w_aligned[i]
        S1 = S1_1w_aligned[i]
        R1 = R1_1w_aligned[i]
        S2 = S2_1w_aligned[i]
        R2 = R2_1w_aligned[i]
        trend_up_1w = price > ema_200_1w_aligned[i]
        
        if position == 1:  # Long
            # Exit: price crosses below weekly S1 or volume drops below average
            if price < S1 or vol_ratio < 1.0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short
            # Exit: price crosses above weekly R1 or volume drops below average
            if price > R1 or vol_ratio < 1.0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
        else:  # Flat
            # Enter long: price crosses above weekly Pivot with volume expansion and uptrend on weekly
            if price > pivot and vol_ratio > 2.0 and trend_up_1w:
                position = 1
                signals[i] = 0.30
            # Enter short: price crosses below weekly Pivot with volume expansion and downtrend on weekly
            elif price < pivot and vol_ratio > 2.0 and not trend_up_1w:
                position = -1
                signals[i] = -0.30
    
    return signals