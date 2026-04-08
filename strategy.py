#!/usr/bin/env python3
"""
6h_1d_trix_volume_regime_v1
Hypothesis: 6-hour strategy combining daily TRIX momentum with volume confirmation and regime filter.
Long when TRIX crosses above zero with volume > 1.5x average and price > daily EMA200 (bullish trend).
Short when TRIX crosses below zero with volume > 1.5x average and price < daily EMA200 (bearish trend).
Exit when TRIX crosses back across zero or volume drops below average.
Uses discrete position sizing (0.25) to minimize churn. Target: 20-30 trades/year.
Works in both bull and bear markets by using momentum (TRIX) and regime filter (EMA200).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_trix_volume_regime_v1"
timeframe = "6h"
leverage = 1.0

def calculate_trix(close, period=15):
    """Calculate TRIX (Triple Exponential Average)"""
    if len(close) < period * 3:
        return np.full_like(close, np.nan, dtype=float)
    
    # First EMA
    ema1 = np.full_like(close, np.nan, dtype=float)
    alpha = 2.0 / (period + 1)
    ema1[period-1] = np.mean(close[:period])
    for i in range(period, len(close)):
        ema1[i] = alpha * close[i] + (1 - alpha) * ema1[i-1]
    
    # Second EMA
    ema2 = np.full_like(close, np.nan, dtype=float)
    ema2[period-1] = np.mean(ema1[:period])
    for i in range(period, len(close)):
        ema2[i] = alpha * ema1[i] + (1 - alpha) * ema2[i-1]
    
    # Third EMA
    ema3 = np.full_like(close, np.nan, dtype=float)
    ema3[period-1] = np.mean(ema2[:period])
    for i in range(period, len(close)):
        ema3[i] = alpha * ema2[i] + (1 - alpha) * ema3[i-1]
    
    # TRIX = (EMA3[i] - EMA3[i-1]) / EMA3[i-1] * 100
    trix = np.full_like(close, np.nan, dtype=float)
    for i in range(period, len(close)):
        if ema3[i-1] != 0:
            trix[i] = (ema3[i] - ema3[i-1]) / ema3[i-1] * 100
    
    return trix

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
    
    # Get daily data for context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily TRIX
    close_1d = df_1d['close'].values
    trix_1d = calculate_trix(close_1d, 15)
    
    # Calculate daily EMA for trend filter
    ema_200_1d = calculate_ema(close_1d, 200)
    
    # Align indicators to 6-hour timeframe
    trix_1d_aligned = align_htf_to_ltf(prices, df_1d, trix_1d)
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Volume confirmation: 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(trix_1d_aligned[i]) or np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        price = close[i]
        trix = trix_1d_aligned[i]
        trix_prev = trix_1d_aligned[i-1] if i > 0 else 0
        ema_200 = ema_200_1d_aligned[i]
        
        if position == 1:  # Long
            # Exit: TRIX crosses below zero or volume drops below average
            if trix < 0 or vol_ratio < 1.0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: TRIX crosses above zero or volume drops below average
            if trix > 0 or vol_ratio < 1.0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: TRIX crosses above zero with volume expansion and uptrend on daily
            if trix > 0 and trix_prev <= 0 and vol_ratio > 1.5 and price > ema_200:
                position = 1
                signals[i] = 0.25
            # Enter short: TRIX crosses below zero with volume expansion and downtrend on daily
            elif trix < 0 and trix_prev >= 0 and vol_ratio > 1.5 and price < ema_200:
                position = -1
                signals[i] = -0.25
    
    return signals