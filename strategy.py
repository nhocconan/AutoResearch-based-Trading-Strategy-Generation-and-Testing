#!/usr/bin/env python3
"""
1d_1w_camarilla_pivot_v1
Hypothesis: Daily strategy using weekly Camarilla pivot with volume confirmation and RSI filter.
Long when price breaks above weekly R2 with volume > 1.5x average and RSI < 70.
Short when price breaks below weekly S2 with volume > 1.5x average and RSI > 30.
Exit when price crosses opposite weekly level.
Target: 10-25 trades/year per symbol to minimize fee drag while capturing meaningful moves.
Works in both bull (breakouts) and bear (breakdowns) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_camarilla_pivot_v1"
timeframe = "1d"
leverage = 1.0

def calculate_rsi(close, period=14):
    """Calculate RSI with proper handling"""
    if len(close) < period + 1:
        return np.full_like(close, np.nan, dtype=float)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    
    avg_gain[period] = np.mean(gain[:period])
    avg_loss[period] = np.mean(loss[:period])
    
    for i in range(period + 1, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i-1]) / period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = np.full_like(close, 100.0, dtype=float)
    rsi = np.where(avg_loss != 0, 100 - (100 / (1 + rs)), 100)
    rsi[avg_loss == 0] = 100
    rsi[avg_gain == 0] = 0
    return rsi

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for context
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly Pivot (using previous weekly bar's data)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    range_1w = high_1w - low_1w
    # Weekly support/resistance levels (Camarilla S2/R2)
    S2_1w = pivot_1w - (range_1w * 1.1 / 6)  # weekly S2
    R2_1w = pivot_1w + (range_1w * 1.1 / 6)  # weekly R2
    
    # Align indicators to daily timeframe
    S2_1w_aligned = align_htf_to_ltf(prices, df_1w, S2_1w)
    R2_1w_aligned = align_htf_to_ltf(prices, df_1w, R2_1w)
    
    # Volume confirmation: 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # RSI filter
    rsi = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(S2_1w_aligned[i]) or np.isnan(R2_1w_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(rsi[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        price = close[i]
        S2 = S2_1w_aligned[i]
        R2 = R2_1w_aligned[i]
        
        if position == 1:  # Long
            # Exit: price crosses below weekly S2
            if price < S2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price crosses above weekly R2
            if price > R2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above weekly R2 with volume expansion and RSI not overbought
            if price > R2 and vol_ratio > 1.5 and rsi[i] < 70:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below weekly S2 with volume expansion and RSI not oversold
            elif price < S2 and vol_ratio > 1.5 and rsi[i] > 30:
                position = -1
                signals[i] = -0.25
    
    return signals