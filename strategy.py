#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_12511_6d_camarilla1d_v2"
timeframe = "6h"
leverage = 1.0

# Parameters
CAMARILLA_MULTIPLIER = 1.1
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels"""
    pivot = (high + low + close) / 3.0
    range_val = high - low
    r4 = pivot + (range_val * CAMARILLA_MULTIPLIER * 1.5)
    r3 = pivot + (range_val * CAMARILLA_MULTIPLIER * 1.25)
    r2 = pivot + (range_val * CAMARILLA_MULTIPLIER * 1.166)
    r1 = pivot + (range_val * CAMARILLA_MULTIPLIER * 1.083)
    s1 = pivot - (range_val * CAMARILLA_MULTIPLIER * 1.083)
    s2 = pivot - (range_val * CAMARILLA_MULTIPLIER * 1.166)
    s3 = pivot - (range_val * CAMARILLA_MULTIPLIER * 1.25)
    s4 = pivot - (range_val * CAMARILLA_MULTIPLIER * 1.5)
    return r4, r3, r2, r1, pivot, s1, s2, s3, s4

def calculate_atr(high, low, close, period):
    """Calculate ATR"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    r4_1d, r3_1d, r2_1d, r1_1d, pivot_1d, s1_1d, s2_1d, s3_1d, s4_1d = calculate_camarilla(
        high_1d, low_1d, close_1d)
    
    # Align all levels to 6h timeframe
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    r2_1d_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    s2_1d_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period (need at least 1 day of data)
    start = 4  # 4x 6h bars = 1 day
    
    for i in range(start, n):
        # Skip if daily Camarilla levels not available
        if np.isnan(r4_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Fade at R3/S3, breakout continuation at R4/S4
        near_r3 = abs(close[i] - r3_1d_aligned[i]) < (0.5 * atr[i])
        near_s3 = abs(close[i] - s3_1d_aligned[i]) < (0.5 * atr[i])
        breakout_r4 = close[i] > r4_1d_aligned[i]
        breakdown_s4 = close[i] < s4_1d_aligned[i]
        
        # Entry conditions
        long_entry = near_s3 or breakout_r4
        short_entry = near_r3 or breakdown_s4
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals