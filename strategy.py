#!/usr/bin/env python3
"""
12h_1w1d_atr_breakout_v1
Hypothesis: 12-hour strategy using weekly ATR breakout with volume confirmation and daily trend filter.
Long when price breaks above weekly ATR-based upper band with volume > 2x average and price > daily EMA50 (bullish trend).
Short when price breaks below weekly ATR-based lower band with volume > 2x average and price < daily EMA50 (bearish trend).
Exit when price returns to weekly midline or volume drops below average.
Uses discrete position sizing (0.25) to minimize churn. Target: 15-25 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w1d_atr_breakout_v1"
timeframe = "12h"
leverage = 1.0

def calculate_atr(high, low, close, period):
    """Calculate ATR with proper handling"""
    if len(high) < period:
        return np.full_like(high, np.nan, dtype=float)
    
    tr = np.zeros(len(high))
    tr[0] = high[0] - low[0]
    for i in range(1, len(high)):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = np.zeros(len(high))
    atr[period-1] = np.mean(tr[:period])
    for i in range(period, len(high)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    return atr

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
    
    # Get weekly data for ATR and midline
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate weekly ATR and midline (using previous week's data)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    atr_14 = calculate_atr(high_1w, low_1w, close_1w, 14)
    midline = (high_1w + low_1w) / 2.0  # Weekly midline (average of high and low)
    
    # Weekly ATR-based bands
    upper_band = midline + (atr_14 * 1.5)
    lower_band = midline - (atr_14 * 1.5)
    
    # Calculate daily EMA for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = calculate_ema(close_1d, 50)
    
    # Align indicators to 12-hour timeframe
    midline_aligned = align_htf_to_ltf(prices, df_1w, midline)
    upper_band_aligned = align_htf_to_ltf(prices, df_1w, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_1w, lower_band)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: 30-period average
    vol_ma = np.full(n, np.nan)
    for i in range(30, n):
        vol_ma[i] = np.mean(volume[i-30:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(midline_aligned[i]) or np.isnan(upper_band_aligned[i]) or 
            np.isnan(lower_band_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        price = close[i]
        midline_val = midline_aligned[i]
        upper = upper_band_aligned[i]
        lower = lower_band_aligned[i]
        trend_up_1d = price > ema_50_1d_aligned[i]
        
        if position == 1:  # Long
            # Exit: price returns to midline or volume drops below average
            if price <= midline_val or vol_ratio < 1.0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price returns to midline or volume drops below average
            if price >= midline_val or vol_ratio < 1.0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above upper band with volume expansion and uptrend on daily
            if price > upper and vol_ratio > 2.0 and trend_up_1d:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below lower band with volume expansion and downtrend on daily
            elif price < lower and vol_ratio > 2.0 and not trend_up_1d:
                position = -1
                signals[i] = -0.25
    
    return signals