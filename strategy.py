#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_12859_6h_1d_supertrend_hl_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
SUPERTREND_PERIOD = 10
SUPERTREND_MULTIPLIER = 3.0
ATR_PERIOD = 10
SIGNAL_SIZE = 0.25
MAX_HOLD_BARS = 48  # Max 12 days (48 * 6h)

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_supertrend(high, low, close, period, multiplier):
    """Calculate Supertrend indicator"""
    atr = calculate_atr(high, low, close, period)
    
    # Calculate upper and lower bands
    upperband = (high + low) / 2 + multiplier * atr
    lowerband = (high + low) / 2 - multiplier * atr
    
    # Initialize supertrend
    supertrend = np.zeros(len(close))
    direction = np.ones(len(close))  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(close)):
        if close[i] > upperband[i-1]:
            direction[i] = 1
        elif close[i] < lowerband[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
            
            if direction[i] == 1 and lowerband[i] < lowerband[i-1]:
                lowerband[i] = lowerband[i-1]
            if direction[i] == -1 and upperband[i] > upperband[i-1]:
                upperband[i] = upperband[i-1]
        
        if direction[i] == 1:
            supertrend[i] = lowerband[i]
        else:
            supertrend[i] = upperband[i]
    
    return supertrend, direction

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate daily Supertrend
    high_d = df_daily['high'].values
    low_d = df_daily['low'].values
    close_d = df_daily['close'].values
    
    supertrend_d, direction_d = calculate_supertrend(high_d, low_d, close_d, SUPERTREND_PERIOD, SUPERTREND_MULTIPLIER)
    
    # Align to 6h timeframe
    supertrend_aligned = align_htf_to_ltf(prices, df_daily, supertrend_d)
    direction_aligned = align_htf_to_ltf(prices, df_daily, direction_d)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = max(SUPERTREND_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if daily supertrend not available
        if np.isnan(supertrend_aligned[i]) or np.isnan(direction_aligned[i]):
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
                bars_since_entry = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
        
        # Time-based exit to prevent overtrading
        if bars_since_entry >= MAX_HOLD_BARS:
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
            continue
        
        # Trend-following signals based on daily supertrend
        long_signal = direction_aligned[i] == 1 and close[i] > supertrend_aligned[i]
        short_signal = direction_aligned[i] == -1 and close[i] < supertrend_aligned[i]
        
        # Generate signals
        if position == 0:
            if long_signal:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.0 * atr[i])  # 2*ATR stop loss
                bars_since_entry = 0
            elif short_signal:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.0 * atr[i])  # 2*ATR stop loss
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals