#!/usr/bin/env python3
"""
Experiment #10855: 6h Donchian Breakout with Weekly Pivot Direction and Volume Confirmation
Hypothesis: 6-hour Donchian(20) breakouts in the direction of weekly trend (price above/below weekly pivot),
confirmed by volume spikes, provide high-probability trades. Weekly pivot acts as dynamic support/resistance.
Designed to work in bull markets (breakouts above pivot) and bear markets (breakdowns below pivot).
Target: 100-200 total trades over 4 years (25-50/year) on 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_10855_6h_donchian20_weekly_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
PIVOT_LOOKBACK = 5  # days for weekly pivot calculation

def calculate_donchian(high, low, period):
    """Calculate Donchian channels"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_atr(high, low, close, period):
    """Calculate ATR"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_weekly_pivot(high, low, close):
    """Calculate weekly pivot point (standard: (H+L+C)/3)"""
    # For weekly data, we calculate pivot from weekly OHLC
    pivot = (high + low + close) / 3.0
    return pivot

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot from weekly OHLC
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    weekly_pivot = calculate_weekly_pivot(weekly_high, weekly_low, weekly_close)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_weekly, weekly_pivot)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    upper, lower = calculate_donchian(high, low, DONCHIAN_PERIOD)
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # Session filter: 00-23 UTC (always on for 6h - reduced frequency already filters noise)
    # Using all hours since 6h has lower frequency
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if weekly pivot not available
        if np.isnan(weekly_pivot_aligned[i]):
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
        
        # Breakout conditions
        breakout_up = (not np.isnan(upper[i-1]) and close[i] > upper[i-1])
        breakout_down = (not np.isnan(lower[i-1]) and close[i] < lower[i-1])
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Weekly pivot position
        above_pivot = close[i] > weekly_pivot_aligned[i]
        below_pivot = close[i] < weekly_pivot_aligned[i]
        
        # Entry conditions: breakout in direction of weekly pivot
        long_entry = breakout_up and volume_ok and above_pivot
        short_entry = breakout_down and volume_ok and below_pivot
        
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