#!/usr/bin/env python3
"""
1d_1w_ema_trend_v1
Hypothesis: 1-day strategy using 1-week EMA trend filter with volume confirmation.
Long when price > 1-week EMA200 and volume > 1.5x average.
Short when price < 1-week EMA200 and volume > 1.5x average.
Exit when price crosses 1-week EMA200.
Uses higher timeframe trend for better signal quality in both bull and bear markets.
Target: 10-25 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_ema_trend_v1"
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
    volume = prices['volume'].values
    
    # Get 1-week data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    # Calculate 1w EMA200 for trend filter
    ema_200_1w = calculate_ema(df_1w['close'].values, 200)
    
    # Align EMA to daily timeframe
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Volume confirmation: 20-day average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):
        # Skip if data not ready
        if np.isnan(ema_200_1w_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        price = close[i]
        ema = ema_200_1w_aligned[i]
        
        if position == 1:  # Long
            # Exit: price crosses below 1w EMA200
            if price < ema:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price crosses above 1w EMA200
            if price > ema:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price above 1w EMA200 with volume expansion
            if price > ema and vol_ratio > 1.5:
                position = 1
                signals[i] = 0.25
            # Enter short: price below 1w EMA200 with volume expansion
            elif price < ema and vol_ratio > 1.5:
                position = -1
                signals[i] = -0.25
    
    return signals