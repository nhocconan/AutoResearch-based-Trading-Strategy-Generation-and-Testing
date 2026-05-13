#!/usr/bin/env python3
"""
1d_Pivot_Price_Action_Strategy
Hypothesis: Daily price action at key weekly pivot points (R1/S1) with volume confirmation 
and volatility filter works in both bull and bear markets. Uses weekly pivots as 
structural support/resistance, enters on rejection or breakout with volume spike.
Exit on opposite pivot touch or volatility expansion. Target: 10-25 trades/year.
"""

name = "1d_Pivot_Price_Action_Strategy"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_true_range(high, low, close):
    """Calculate True Range"""
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[0.0], np.maximum(tr1, np.maximum(tr2, tr3))])
    return tr

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr = calculate_true_range(high, low, close)
    atr = np.zeros_like(close, dtype=float)
    if len(close) < period:
        return atr
    # First ATR is simple average
    atr[period-1] = np.mean(tr[1:period])
    # Wilder's smoothing: ATR[t] = (ATR[t-1] * (period-1) + TR[t]) / period
    for i in range(period, len(close)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    return atr

def calculate_pivot_points(high, low, close):
    """Calculate standard pivot points: P = (H+L+C)/3, R1 = 2P - L, S1 = 2P - H"""
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
    
    # Calculate ATR for volatility filter (14-period)
    atr_14 = calculate_atr(high, low, close, 14)
    atr_ma = np.zeros(n)
    for i in range(14, n):
        atr_ma[i] = np.mean(atr_14[i-14:i])
    # Low volatility filter: avoid choppy markets
    low_vol = atr_14 < 0.8 * atr_ma
    
    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points
    wp_high = df_1w['high'].values
    wp_low = df_1w['low'].values
    wp_close = df_1w['close'].values
    
    wp_pivot, wp_r1, wp_s1 = calculate_pivot_points(wp_high, wp_low, wp_close)
    
    # Align weekly pivots to daily timeframe
    wp_pivot_aligned = align_htf_to_ltf(prices, df_1w, wp_pivot)
    wp_r1_aligned = align_htf_to_ltf(prices, df_1w, wp_r1)
    wp_s1_aligned = align_htf_to_ltf(prices, df_1w, wp_s1)
    
    # Volume confirmation: volume > 1.5 * 20-day average
    vol_ma = np.zeros(n)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_conf = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after warmup period
        # Get current values
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        r1 = wp_r1_aligned[i]
        s1 = wp_s1_aligned[i]
        vol_ok = volume_conf[i]
        vol_filter = low_vol[i]
        
        if position == 0:
            # LONG ENTRY: Price rejects S1 with bullish close and volume
            # Conditions: low touches/goes below S1, closes back above S1, volume spike
            if curr_low <= s1 and curr_close > s1 and vol_ok and vol_filter:
                signals[i] = 0.25
                position = 1
            # SHORT ENTRY: Price rejects R1 with bearish close and volume
            # Conditions: high touches/goes above R1, closes back below R1, volume spike
            elif curr_high >= r1 and curr_close < r1 and vol_ok and vol_filter:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reaches R1 or volatility expands
            if curr_high >= r1 or not vol_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reaches S1 or volatility expands
            if curr_low <= s1 or not vol_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals