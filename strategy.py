#!/usr/bin/env python3
"""
6h_1w_1d_Weekly_Pivot_Directional
Hypothesis: Weekly pivot levels act as strong support/resistance in trending markets. 
We use daily trend direction (close > EMA50) to determine bias: long when above weekly pivot + daily uptrend, short when below weekly pivot + daily downtrend.
Weekly pivots calculated from prior week's OHLC. Entry only when price touches pivot level with rejection (wick rejection) and volume confirmation.
Works in bull markets by buying dips to weekly support in uptrend, works in bear markets by selling rallies to weekly resistance in downtrend.
Target: 15-30 trades/year on 6h (60-120 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_pivot_points(high, low, close):
    """Calculate standard pivot points: P = (H+L+C)/3, R1 = 2P-L, S1 = 2P-H"""
    pivot = (high + low + close) / 3.0
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    return pivot, r1, s1

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points (prior week's OHLC)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points from prior week
    pivot_1w, r1_1w, s1_1w = calculate_pivot_points(high_1w, low_1w, close_1w)
    # Arrays of same length as weekly data
    pivot_1w_arr = np.full_like(high_1w, pivot_1w)
    r1_1w_arr = np.full_like(high_1w, r1_1w)
    s1_1w_arr = np.full_like(high_1w, s1_1w)
    
    # Get daily data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Daily EMA50 for trend direction
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly pivot levels to 6h timeframe
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w_arr)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w_arr)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w_arr)
    
    # Align daily EMA50 to 6h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if any data not ready
        if (np.isnan(pivot_1w_aligned[i]) or np.isnan(r1_1w_aligned[i]) or 
            np.isnan(s1_1w_aligned[i]) or np.isnan(ema50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Daily trend direction: 1 for uptrend (close > EMA50), -1 for downtrend (close < EMA50)
        daily_trend = 1 if close[i] > ema50_1d_aligned[i] else -1
        
        # Price action: check for rejection at pivot levels
        # Bullish rejection: long wick below support (low touches S1, close > S1)
        bullish_rejection = (low[i] <= s1_1w_aligned[i] * 1.001) and (close[i] > s1_1w_aligned[i])
        # Bearish rejection: long wick above resistance (high touches R1, close < R1)
        bearish_rejection = (high[i] >= r1_1w_aligned[i] * 0.999) and (close[i] < r1_1w_aligned[i])
        
        # Volume confirmation: current volume > 1.2x 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
            volume_confirm = volume[i] > (vol_ma_20 * 1.2)
        else:
            volume_confirm = False
        
        # Entry logic
        if daily_trend == 1 and bullish_rejection and volume_confirm:
            # Uptrend + bullish rejection at weekly support = long
            if position != 1:
                position = 1
                signals[i] = position_size
            else:
                signals[i] = position_size
        elif daily_trend == -1 and bearish_rejection and volume_confirm:
            # Downtrend + bearish rejection at weekly resistance = short
            if position != -1:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = -position_size
        else:
            # Hold current position or flat
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1w_1d_Weekly_Pivot_Directional"
timeframe = "6h"
leverage = 1.0