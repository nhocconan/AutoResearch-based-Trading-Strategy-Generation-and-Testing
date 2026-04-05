#!/usr/bin/env python3
"""
Experiment #9959: 6h/12h Donchian Breakout + Pivot Range + Volume Spike
Hypothesis: Combining 6h Donchian breakouts with 12h pivot range (CPR) creates
high-probability trades by filtering for breakouts that escape consolidation.
Works in bull/bear: breaks above CPR high = long, breaks below CPR low = short.
Volume spike confirms breakout strength. Target: 100-200 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_9959_6h_12h_donchian_pivot_range_volume_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOLUME_SPIKE_MULTIPLIER = 1.8
CPR_LOOKBACK = 10
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_donchian_channels(high, low, period):
    """Calculate Donchian channels"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_cpr(high, low, close, lookback):
    """Calculate Central Pivot Range (CPR)"""
    pivot = (high + low + close) / 3.0
    bc = (high + low) / 2.0
    tc = (pivot - bc) + pivot
    # Rolling CPR values
    tc_series = pd.Series(tc)
    bc_series = pd.Series(bc)
    tc_val = tc_series.rolling(window=lookback, min_periods=lookback).max().values
    bc_val = bc_series.rolling(window=lookback, min_periods=lookback).min().values
    return tc_val, bc_val  # TC = top of CPR, BC = bottom of CPR

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

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
    
    # Load 12h data ONCE before loop for CPR calculation
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h CPR for pivot range
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    cpr_top, cpr_bottom = calculate_cpr(high_12h, low_12h, close_12h, CPR_LOOKBACK)
    
    # Align CPR to 6h timeframe
    cpr_top_aligned = align_htf_to_ltf(prices, df_12h, cpr_top)
    cpr_bottom_aligned = align_htf_to_ltf(prices, df_12h, cpr_bottom)
    
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
    start = max(DONCHIAN_PERIOD, CPR_LOOKBACK, 20) + 1
    
    for i in range(start, n):
        # Skip if CPR not available
        if np.isnan(cpr_top_aligned[i]) or np.isnan(cpr_bottom_aligned[i]):
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
        
        # Breakout conditions relative to CPR
        above_cpr_top = close[i] > cpr_top_aligned[i] if not np.isnan(cpr_top_aligned[i]) else False
        below_cpr_bottom = close[i] < cpr_bottom_aligned[i] if not np.isnan(cpr_bottom_aligned[i]) else False
        
        # Donchian breakout
        bullish_breakout = close[i] > donch_upper[i] if not np.isnan(donch_upper[i]) else False
        bearish_breakout = close[i] < donch_lower[i] if not np.isnan(donch_lower[i]) else False
        
        # Entry: Donchian breakout in direction of CPR breakout with volume
        long_entry = bullish_breakout and above_cpr_top and volume_spike
        short_entry = bearish_breakout and below_cpr_bottom and volume_spike
        
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