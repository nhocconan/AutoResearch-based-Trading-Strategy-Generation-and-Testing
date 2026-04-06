#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_12491_6d_camarilla1d_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
CAMARILLA_MULT = 1.1
SIGNAL_SIZE = 0.28
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
MIN_HOLD_BARS = 3

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate ATR"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels"""
    pivot = (high + low + close) / 3
    range_hl = high - low
    r4 = pivot + (range_hl * CAMARILLA_MULT * 1.5)
    r3 = pivot + (range_hl * CAMARILLA_MULT * 1.25)
    s3 = pivot - (range_hl * CAMARILLA_MULT * 1.25)
    s4 = pivot - (range_hl * CAMARILLA_MULT * 1.5)
    return r3, r4, s3, s4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily Camarilla levels
    r3_1d, r4_1d, s3_1d, s4_1d = calculate_camarilla(
        df_1d['high'].values,
        df_1d['low'].values,
        df_1d['close'].values
    )
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
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
    bars_held = 0
    
    # Start from warmup period
    start = ATR_PERIOD + 1
    
    for i in range(start, n):
        # Skip if daily Camarilla not available
        if np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]):
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
                bars_held = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                bars_held = 0
                continue
        
        # Minimum hold period
        if position != 0:
            bars_held += 1
            if bars_held < MIN_HOLD_BARS:
                signals[i] = position * SIGNAL_SIZE
                continue
        
        # Camarilla breakout conditions
        long_breakout = close[i] > r4_1d_aligned[i-1]  # break above R4
        short_breakout = close[i] < s4_1d_aligned[i-1]  # break below S4
        
        # Camarilla fade conditions
        long_fade = close[i] < s3_1d_aligned[i] and close[i] > s4_1d_aligned[i]
        short_fade = close[i] > r3_1d_aligned[i] and close[i] < r4_1d_aligned[i]
        
        # Entry conditions
        long_entry = long_breakout or long_fade
        short_entry = short_breakout or short_fade
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
                bars_held = 0
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
                bars_held = 0
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals