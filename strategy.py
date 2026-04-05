#!/usr/bin/env python3
"""
Experiment #9227: 6h Donchian breakout + weekly pivot direction + volume confirmation
Hypothesis: Donchian breakouts capture trends; weekly pivot (from 1w) defines long-term bias; volume confirms institutional participation.
Targets 75-200 total trades over 4 years (19-50/year) to balance opportunity and cost. Works in bull (breakouts) and bear (filtered shorts).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_9227_6h_donchian20_1w_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.8
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.2

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_pivot_points(high, low, close):
    """Calculate weekly pivot points: P, R1, S1, R2, S2, R3, S3, R4, S4"""
    pivot = (high + low + close) / 3.0
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    r3 = high + 2 * (pivot - low)
    s3 = low - 2 * (high - pivot)
    r4 = r3 + (high - low)
    s4 = s3 - (high - low)
    return pivot, r1, s1, r2, s2, r3, s3, r4, s4

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Initialize arrays for pivot levels
    pivot_vals = np.full_like(close_1w, np.nan)
    r3_vals = np.full_like(close_1w, np.nan)
    s3_vals = np.full_like(close_1w, np.nan)
    r4_vals = np.full_like(close_1w, np.nan)
    s4_vals = np.full_like(close_1w, np.nan)
    
    # Calculate pivot points for each weekly bar
    for i in range(len(close_1w)):
        if not (np.isnan(high_1w[i]) or np.isnan(low_1w[i]) or np.isnan(close_1w[i])):
            pivot, r1, s1, r2, s2, r3, s3, r4, s4 = calculate_pivot_points(high_1w[i], low_1w[i], close_1w[i])
            pivot_vals[i] = pivot
            r3_vals[i] = r3
            s3_vals[i] = s3
            r4_vals[i] = r4
            s4_vals[i] = s4
    
    # Align to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot_vals)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3_vals)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3_vals)
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4_vals)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4_vals)
    
    # Calculate LTF indicators (6h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels
    donchian_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    donchian_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(pivot_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
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
        
        # Determine market bias from weekly pivot
        # Long bias: price above weekly R3 (bullish territory)
        # Short bias: price below weekly S3 (bearish territory)
        long_bias = close[i] > r3_aligned[i] if not np.isnan(r3_aligned[i]) else False
        short_bias = close[i] < s3_aligned[i] if not np.isnan(s3_aligned[i]) else False
        
        # Donchian breakout conditions
        long_breakout = close[i] > donchian_high[i-1]  # Break above previous period's high
        short_breakout = close[i] < donchian_low[i-1]  # Break below previous period's low
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Entry conditions
        long_entry = long_bias and long_breakout and volume_confirmed
        short_entry = short_bias and short_breakout and volume_confirmed
        
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