#!/usr/bin/env python3
"""
Experiment #10171: 6h Donchian Breakout + Daily Pivot + Volume Spike
Hypothesis: Donchian(20) breakouts in the direction of daily pivot (R1/S1) with volume confirmation
provide high-probability trend continuation. Works in bull markets (breakouts above daily pivot)
and bear markets (breakdowns below daily pivot). Uses 1d pivot levels for trend filtering and
volume spike for confirmation. Target: 75-150 total trades over 4 years (19-38/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf

name = "exp_10171_6h_donchian_breakout_daily_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOLUME_SPIKE_MULTIPLIER = 1.8
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_donchian_channels(high, low, period):
    """Calculate Donchian channels"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_pivot_points(high, low, close):
    """Calculate daily pivot points"""
    pivot = (high + low + close) / 3.0
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    return pivot, r1, s1, r2, s2

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data ONCE before loop for pivot levels
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate daily pivot points
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    daily_close = df_daily['close'].values
    pivot, r1, s1, r2, s2 = calculate_pivot_points(daily_high, daily_low, daily_close)
    
    # Align daily pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_daily, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_daily, r1)
    s1_aligned = align_htf_to_ltf(prices, df_daily, s1)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels
    donch_upper, donch_lower = calculate_donchian_channels(high, low, DONCHIAN_PERIOD)
    
    # Volume moving average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, 20) + 1
    
    for i in range(start, n):
        # Skip if daily pivot not available
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
        
        # Volume spike confirmation
        volume_spike = volume[i] > (volume_ma[i] * VOLUME_SPIKE_MULTIPLIER) if not np.isnan(volume_ma[i]) else False
        
        # Pivot filter: price above/below daily pivot (S1/R1)
        above_pivot = close[i] > pivot_aligned[i]
        below_pivot = close[i] < pivot_aligned[i]
        above_r1 = close[i] > r1_aligned[i] if not np.isnan(r1_aligned[i]) else False
        below_s1 = close[i] < s1_aligned[i] if not np.isnan(s1_aligned[i]) else False
        
        # Breakout conditions
        bullish_breakout = close[i] > donch_upper[i] if not np.isnan(donch_upper[i]) else False
        bearish_breakout = close[i] < donch_lower[i] if not np.isnan(donch_lower[i]) else False
        
        # Entry conditions: breakout in direction of pivot with volume
        long_entry = bullish_breakout and above_pivot and volume_spike
        short_entry = bearish_breakout and below_pivot and volume_spike
        
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

def align_htf_to_ltf(prices, df_htf, values):
    """Align HTF values to LTF with proper shift(1)"""
    from mtf_data import align_htf_to_ltf
    return align_htf_to_ltf(prices, df_htf, values)