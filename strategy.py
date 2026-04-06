#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_13919_6d_camarilla1d_pivot_fade_v2"
timeframe = "6h"
leverage = 1.0

# Hypothesis: 6h Camarilla pivot reversal from 1d pivots with volume confirmation
# Fade at R3/S3 levels, breakout continuation at R4/S4
# Works in sideways markets (reversion) and strong trends (breakout)
# Target: 80-150 trades over 4 years by requiring confluence of price at pivot + volume spike

def calculate_camarilla_pivot(high, low, close):
    """Calculate Camarilla pivot levels for the day"""
    pivot = (high + low + close) / 3.0
    range_val = high - low
    r4 = close + (range_val * 1.1 / 2)
    r3 = close + (range_val * 1.1 / 4)
    r2 = close + (range_val * 1.1 / 6)
    r1 = close + (range_val * 1.1 / 12)
    s1 = close - (range_val * 1.1 / 12)
    s2 = close - (range_val * 1.1 / 6)
    s3 = close - (range_val * 1.1 / 4)
    s4 = close - (range_val * 1.1 / 2)
    return r4, r3, r2, r1, pivot, s1, s2, s3, s4

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for Camarilla pivots ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Camarilla pivots
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivots for each day
    r4_1d = np.full_like(close_1d, np.nan)
    r3_1d = np.full_like(close_1d, np.nan)
    r2_1d = np.full_like(close_1d, np.nan)
    r1_1d = np.full_like(close_1d, np.nan)
    pivot_1d = np.full_like(close_1d, np.nan)
    s1_1d = np.full_like(close_1d, np.nan)
    s2_1d = np.full_like(close_1d, np.nan)
    s3_1d = np.full_like(close_1d, np.nan)
    s4_1d = np.full_like(close_1d, np.nan)
    
    for i in range(len(close_1d)):
        r4, r3, r2, r1, p, s1, s2, s3, s4 = calculate_camarilla_pivot(high_1d[i], low_1d[i], close_1d[i])
        r4_1d[i] = r4
        r3_1d[i] = r3
        r2_1d[i] = r2
        r1_1d[i] = r1
        pivot_1d[i] = p
        s1_1d[i] = s1
        s2_1d[i] = s2
        s3_1d[i] = s3
        s4_1d[i] = s4
    
    # Align pivots to 6h timeframe
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    r2_1d_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    s2_1d_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # 6h data for price and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(20, 1) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or 
            np.isnan(r4_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check stops
        if position == 1:  # long position
            # Check stop loss
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # short position
            # Check stop loss
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Volume confirmation - spike > 2.0x average
        volume_ok = volume[i] > (volume_ma[i] * 2.0)
        
        # Price proximity to pivot levels (within 0.1% for fade, 0.05% for breakout)
        proximity_threshold = 0.001  # 0.1%
        breakout_threshold = 0.0005  # 0.05%
        
        near_r3 = abs(close[i] - r3_1d_aligned[i]) / r3_1d_aligned[i] < proximity_threshold
        near_s3 = abs(close[i] - s3_1d_aligned[i]) / s3_1d_aligned[i] < proximity_threshold
        near_r4 = abs(close[i] - r4_1d_aligned[i]) / r4_1d_aligned[i] < breakout_threshold
        near_s4 = abs(close[i] - s4_1d_aligned[i]) / s4_1d_aligned[i] < breakout_threshold
        
        # Fade at R3/S3, breakout at R4/S4
        fade_at_r3 = near_r3 and close[i] < r3_1d_aligned[i]  # Price at R3 and rejecting
        fade_at_s3 = near_s3 and close[i] > s3_1d_aligned[i]  # Price at S3 and bouncing
        breakout_at_r4 = near_r4 and close[i] > r4_1d_aligned[i]  # Breaking above R4
        breakdown_at_s4 = near_s4 and close[i] < s4_1d_aligned[i]  # Breaking below S4
        
        # Entry signals
        long_signal = volume_ok and (fade_at_s3 or breakout_at_r4)
        short_signal = volume_ok and (fade_at_r3 or breakdown_at_s4)
        
        # Generate signals
        if position == 0:
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.0 * atr[i])
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.0 * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on fade at R3 or stop loss
            if near_r3 and close[i] < r3_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short on fade at S3 or stop loss
            if near_s3 and close[i] > s3_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals