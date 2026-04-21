#!/usr/bin/env python3
"""
4h_PivotPoint_R1S1_Breakout_Volume_Confirmation_V1
Hypothesis: Price breaking above/below daily Camarilla pivot R1/S1 with volume > 2x 20-period average and aligned with daily EMA21 direction yields high-probability breakouts. Uses daily EMA21 as trend filter to avoid counter-trend trades. Targets 20-40 trades/year with strict entry conditions to minimize fee drag. Works in bull/bear markets by following the daily trend direction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ema(close, period):
    """Calculate Exponential Moving Average"""
    ema = np.zeros_like(close)
    if len(close) < period:
        return ema
    multiplier = 2 / (period + 1)
    ema[0] = close[0]
    for i in range(1, len(close)):
        ema[i] = (close[i] - ema[i-1]) * multiplier + ema[i-1]
    return ema

def calculate_camarilla_pivots(high, low, close):
    """Calculate Camarilla pivot levels"""
    pivot = (high + low + close) / 3
    range_val = high - low
    r1 = close + (range_val * 1.1 / 12)
    s1 = close - (range_val * 1.1 / 12)
    return pivot, r1, s1

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA21 for trend filter
    close_1d = df_1d['close'].values
    ema_21_1d = calculate_ema(close_1d, 21)
    ema_21_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_21_1d)
    
    # Calculate daily Camarilla pivots
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    _, r1_1d, s1_1d = calculate_camarilla_pivots(high_1d, low_1d, close_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if daily indicators not ready
        if np.isnan(ema_21_1d_aligned[i]) or np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation: current volume > 2x 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 2.0 * vol_ma
        else:
            volume_ok = False
        
        # Trend filter: price above/below daily EMA21
        price_above_ema = price > ema_21_1d_aligned[i]
        price_below_ema = price < ema_21_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1 + volume confirmation + price above daily EMA21
            if price > r1_1d_aligned[i] and volume_ok and price_above_ema:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + volume confirmation + price below daily EMA21
            elif price < s1_1d_aligned[i] and volume_ok and price_below_ema:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below S1 or price crosses below daily EMA21
            if price < s1_1d_aligned[i] or price < ema_21_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above R1 or price crosses above daily EMA21
            if price > r1_1d_aligned[i] or price > ema_21_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_PivotPoint_R1S1_Breakout_Volume_Confirmation_V1"
timeframe = "4h"
leverage = 1.0