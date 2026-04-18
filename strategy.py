#!/usr/bin/env python3
"""
1d Weekly ATR Breakout + Volume Confirmation + Trend Filter
Hypothesis: Weekly volatility expansion (ATR breakout) signals institutional participation. Combined with volume surge and trend alignment (price > weekly EMA50), it captures strong trends in both bull and bear markets. Low trade frequency due to strict weekly timeframe and multi-condition entry.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = np.zeros_like(close)
    if len(close) < period:
        return atr
    atr[period-1] = np.mean(tr[:period])
    for i in range(period, len(tr)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    return atr

def calculate_ema(data, period):
    """Calculate Exponential Moving Average"""
    if len(data) < period:
        return np.full_like(data, np.nan)
    ema = np.zeros_like(data)
    ema[0] = data[0]
    alpha = 2 / (period + 1)
    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
    return ema

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for volatility and trend filters
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    # Calculate weekly ATR for volatility breakout
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    atr_weekly = calculate_atr(high_weekly, low_weekly, close_weekly, period=14)
    
    # Calculate weekly EMA50 for trend filter
    ema50_weekly = calculate_ema(close_weekly, period=50)
    
    # Align weekly indicators to daily timeframe
    atr_weekly_aligned = align_htf_to_ltf(prices, df_weekly, atr_weekly)
    ema50_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema50_weekly)
    
    # Volume confirmation: daily volume > 2.0x 20-day average
    vol_ma = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 20:
            vol_ma[i] = np.mean(volume[max(0, i-19):i+1]) if i >= 0 else volume[i]
        else:
            vol_ma[i] = np.mean(volume[i-19:i+1])
    vol_surge = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for indicators
    
    for i in range(start_idx, n):
        if np.isnan(atr_weekly_aligned[i]) or np.isnan(ema50_weekly_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        atr_val = atr_weekly_aligned[i]
        ema50_val = ema50_weekly_aligned[i]
        vol_ok = vol_surge[i]
        
        if position == 0:
            # Enter long: price > weekly EMA50 (uptrend) + weekly ATR breakout + volume surge
            # ATR breakout: today's true range > 1.5x weekly ATR
            tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
            if (close[i] > ema50_val and 
                tr > 1.5 * atr_val and 
                vol_ok):
                signals[i] = 0.25
                position = 1
            # Enter short: price < weekly EMA50 (downtrend) + weekly ATR breakdown + volume surge
            # ATR breakdown: today's true range > 1.5x weekly ATR
            elif (close[i] < ema50_val and 
                  tr > 1.5 * atr_val and 
                  vol_ok):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below weekly EMA50 or volatility contracts
            tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
            if close[i] < ema50_val or tr < 0.8 * atr_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above weekly EMA50 or volatility contracts
            tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
            if close[i] > ema50_val or tr < 0.8 * atr_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Weekly_ATRBreakout_VolumeSurge_TrendFilter"
timeframe = "1d"
leverage = 1.0