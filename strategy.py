#!/usr/bin/env python3
"""
Experiment #9327: 6h Donchian breakout + 1d pivot direction + volume confirmation
Hypothesis: Combines daily pivot levels (1d) for institutional bias with 6h Donchian breakouts for entry timing.
Works in bull markets (breakouts above R1/R2 with bullish pivot) and bear markets (breakdowns below S1/S2 with bearish pivot).
Volume confirmation filters false breakouts. Targets 75-200 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_9327_6h_donchian20_1d_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_pivot_points(high, low, close):
    """Calculate daily pivot points and support/resistance levels"""
    pivot = (high + low + close) / 3.0
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    r3 = high + 2 * (pivot - low)
    s3 = low - 2 * (high - pivot)
    return pivot, r1, r2, r3, s1, s2, s3

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for pivot points)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d pivot points
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot levels for each day
    pivot_vals = np.full_like(close_1d, np.nan)
    r1_vals = np.full_like(close_1d, np.nan)
    r2_vals = np.full_like(close_1d, np.nan)
    r3_vals = np.full_like(close_1d, np.nan)
    s1_vals = np.full_like(close_1d, np.nan)
    s2_vals = np.full_like(close_1d, np.nan)
    s3_vals = np.full_like(close_1d, np.nan)
    
    for i in range(len(close_1d)):
        if not (np.isnan(high_1d[i]) or np.isnan(low_1d[i]) or np.isnan(close_1d[i])):
            pivot, r1, r2, r3, s1, s2, s3 = calculate_pivot_points(high_1d[i], low_1d[i], close_1d[i])
            pivot_vals[i] = pivot
            r1_vals[i] = r1
            r2_vals[i] = r2
            r3_vals[i] = r3
            s1_vals[i] = s1
            s2_vals[i] = s2
            s3_vals[i] = s3
    
    # Align pivot levels to 6h timeframe (using previous day's levels)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_vals)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_vals)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2_vals)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_vals)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_vals)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2_vals)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_vals)
    
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
        
        # Determine market bias from 1d pivot levels
        # Bullish bias: price above pivot (favoring longs)
        # Bearish bias: price below pivot (favoring shorts)
        bull_bias = close[i] > pivot_aligned[i]
        bear_bias = close[i] < pivot_aligned[i]
        
        # Donchian breakout conditions
        long_breakout = close[i] > donchian_high[i-1]  # Break above previous period's high
        short_breakout = close[i] < donchian_low[i-1]  # Break below previous period's low
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Entry conditions - aligned with pivot bias
        long_entry = bull_bias and long_breakout and volume_confirmed
        short_entry = bear_bias and short_breakout and volume_confirmed
        
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