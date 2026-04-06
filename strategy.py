#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_12507_6d_camarilla1d_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
PIVOT_PERIOD = 1  # Daily pivots
CAMARILLA_MULT = 1.1  # For R3/S3, R4/S4
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.8
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
MAX_HOLD_BARS = 48  # Max 8 days (48 * 6h)

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels"""
    pivot = (high + low + close) / 3.0
    range_val = high - low
    r3 = pivot + (range_val * CAMARILLA_MULT * 1.1)
    s3 = pivot - (range_val * CAMARILLA_MULT * 1.1)
    r4 = pivot + (range_val * CAMARILLA_MULT * 1.5)
    s4 = pivot - (range_val * CAMARILLA_MULT * 1.5)
    return r3, s3, r4, s4

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
    
    r3, s3, r4, s4 = calculate_camarilla(high_1d, low_1d, close_1d)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = max(VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if daily Camarilla not available
        if np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Update bars since entry
        if position != 0:
            bars_since_entry += 1
        
        # Check stoploss or max hold
        exit_signal = False
        if position == 1:  # long position
            if close[i] <= stop_price or bars_since_entry >= MAX_HOLD_BARS:
                exit_signal = True
        elif position == -1:  # short position
            if close[i] >= stop_price or bars_since_entry >= MAX_HOLD_BARS:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
            continue
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Fade at R3/S3, breakout at R4/S4
        fade_long = close[i] < s3_aligned[i] and close[i] > s4_aligned[i]
        fade_short = close[i] > r3_aligned[i] and close[i] < r4_aligned[i]
        breakout_long = close[i] > r4_aligned[i]
        breakout_short = close[i] < s4_aligned[i]
        
        # Entry conditions
        long_entry = volume_ok and (fade_long or breakout_long)
        short_entry = volume_ok and (fade_short or breakout_short)
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
                bars_since_entry = 0
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals