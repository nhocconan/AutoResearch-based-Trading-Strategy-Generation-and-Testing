#!/usr/bin/env python3
"""
6h_1d_cci_reversal_v2
Hypothesis: 6-hour strategy using CCI on 1-day for trend direction and RSI on 6-hour for mean reversion entries.
Long when CCI(20) on 1d > 100 (uptrend) and RSI(14) on 6h < 30 (oversold).
Short when CCI(20) on 1d < -100 (downtrend) and RSI(14) on 6h > 70 (overbought).
Exit when RSI crosses back to neutral (40-60 range).
Uses higher timeframe trend filter to avoid counter-trend trades in both bull and bear markets.
Target: 15-40 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_cci_reversal_v2"
timeframe = "6h"
leverage = 1.0

def calculate_cci(high, low, close, period):
    """Calculate Commodity Channel Index"""
    if len(close) < period:
        return np.full_like(close, np.nan, dtype=float)
    
    tp = (high + low + close) / 3.0
    sma = np.full_like(tp, np.nan, dtype=float)
    for i in range(period, len(tp)):
        sma[i] = np.mean(tp[i-period+1:i+1])
    
    mad = np.full_like(tp, np.nan, dtype=float)
    for i in range(period, len(tp)):
        mad[i] = np.mean(np.abs(tp[i-period+1:i+1] - sma[i]))
    
    cci = np.full_like(tp, np.nan, dtype=float)
    for i in range(period, len(tp)):
        if mad[i] != 0:
            cci[i] = (tp[i] - sma[i]) / (0.015 * mad[i])
    return cci

def calculate_rsi(close, period):
    """Calculate Relative Strength Index"""
    if len(close) < period + 1:
        return np.full_like(close, np.nan, dtype=float)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(close, np.nan, dtype=float)
    avg_loss = np.full_like(close, np.nan, dtype=float)
    
    avg_gain[period] = np.mean(gain[:period])
    avg_loss[period] = np.mean(loss[:period])
    
    for i in range(period + 1, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
    
    rs = np.full_like(close, np.nan, dtype=float)
    for i in range(period, len(close)):
        if avg_loss[i] != 0:
            rs[i] = avg_gain[i] / avg_loss[i]
        else:
            rs[i] = np.inf
    
    rsi = np.full_like(close, np.nan, dtype=float)
    for i in range(period, len(close)):
        rsi[i] = 100 - (100 / (1 + rs[i]))
    return rsi

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1-day data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate CCI on 1-day for trend direction
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    cci_20_1d = calculate_cci(high_1d, low_1d, close_1d, 20)
    cci_20_1d_aligned = align_htf_to_ltf(prices, df_1d, cci_20_1d)
    
    # Calculate RSI on 6-hour for entry signals
    rsi_14_6h = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if data not ready
        if np.isnan(cci_20_1d_aligned[i]) or np.isnan(rsi_14_6h[i]):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        cci = cci_20_1d_aligned[i]
        rsi = rsi_14_6h[i]
        
        if position == 1:  # Long
            # Exit: RSI crosses above 40 (exiting oversold)
            if rsi > 40:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: RSI crosses below 60 (exiting overbought)
            if rsi < 60:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: CCI indicates uptrend on 1d and RSI oversold on 6h
            if cci > 100 and rsi < 30:
                position = 1
                signals[i] = 0.25
            # Enter short: CCI indicates downtrend on 1d and RSI overbought on 6h
            elif cci < -100 and rsi > 70:
                position = -1
                signals[i] = -0.25
    
    return signals