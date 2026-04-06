#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_12507_6d_camarilla1d_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
CAMARILLA_MULT = 1.1
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
MIN_CANDLES = 10

def calculate_camarilla(high, low, close):
    """Calculate Camarilla levels (H4/L4 for entry, H3/L3 for stop reference)"""
    range_val = high - low
    # H4 = Close + 1.1 * Range * 1.5
    # L4 = Close - 1.1 * Range * 1.5
    # H3 = Close + 1.1 * Range * 1.25
    # L3 = Close - 1.1 * Range * 1.25
    h4 = close + CAMARILLA_MULT * range_val * 1.5
    l4 = close - CAMARILLA_MULT * range_val * 1.5
    h3 = close + CAMARILLA_MULT * range_val * 1.25
    l3 = close - CAMARILLA_MULT * range_val * 1.25
    return h4, l4, h3, l3

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
    if n < MIN_CANDLES:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    h4_1d, l4_1d, h3_1d, l3_1d = calculate_camarilla(high_1d, low_1d, close_1d)
    h4_1d_aligned = align_htf_to_ltf(prices, df_1d, h4_1d)
    l4_1d_aligned = align_htf_to_ltf(prices, df_1d, l4_1d)
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = 1
    
    for i in range(start, n):
        # Skip if daily levels not available
        if np.isnan(h4_1d_aligned[i]) or np.isnan(l4_1d_aligned[i]):
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
        
        # Fade at H4/L4 with reversal signals
        near_h4 = close[i] >= h4_1d_aligned[i] * 0.999  # near or above H4
        near_l4 = close[i] <= l4_1d_aligned[i] * 1.001  # near or below L4
        
        # Check for rejection (price moving back inside levels)
        reject_high = (close[i] < h4_1d_aligned[i]) and (prices['close'].values[i-1] >= h4_1d_aligned[i-1])
        reject_low = (close[i] > l4_1d_aligned[i]) and (prices['close'].values[i-1] <= l4_1d_aligned[i-1])
        
        # Entry conditions
        long_entry = near_l4 and reject_low
        short_entry = near_h4 and reject_high
        
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