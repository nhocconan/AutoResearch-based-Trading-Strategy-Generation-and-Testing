#!/usr/bin/env python3
"""
Experiment #9539: 6h Donchian Breakout + Weekly Pivot Direction + Volume Confirmation.
Hypothesis: Combine daily Donchian breakouts with weekly pivot direction (bullish/bearish bias)
and volume confirmation to capture strong directional moves in both bull and bear markets.
Weekly pivot provides trend bias, while Donchian breakouts capture momentum.
Targets 75-150 total trades over 4 years (19-38/year) to balance opportunity and cost.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf

name = "exp_9539_6h_donchian_weekly_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOLUME_SPIKE_MULTIPLIER = 2.0
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_donchian(high, low, period):
    """Calculate Donchian channels"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_weekly_pivot(high, low, close):
    """
    Calculate weekly pivot levels from previous week's OHLC
    Pivot = (High + Low + Close) / 3
    Bias: Above pivot = bullish, Below pivot = bearish
    """
    pivot = (high + low + close) / 3
    return pivot

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for Donchian, 1w for weekly pivot)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d Donchian channels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    dc_upper, dc_lower = calculate_donchian(high_1d, low_1d, DONCHIAN_PERIOD)
    
    # Calculate 1w weekly pivot (using previous week's OHLC)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    weekly_pivot = calculate_weekly_pivot(high_1w, low_1w, close_1w)
    
    # Align HTF levels to 6h timeframe
    dc_upper_aligned = align_ltf_to_htf(prices, df_1d, dc_upper)
    dc_lower_aligned = align_ltf_to_htf(prices, df_1d, dc_lower)
    weekly_pivot_aligned = align_ltf_to_htf(prices, df_1w, weekly_pivot)
    
    # Calculate LTF indicators (6h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
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
        # Skip if HTF data not available
        if np.isnan(dc_upper_aligned[i]) or np.isnan(dc_lower_aligned[i]) or np.isnan(weekly_pivot_aligned[i]):
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
        
        # Weekly pivot bias: above = bullish bias, below = bearish bias
        bullish_bias = close[i] > weekly_pivot_aligned[i]
        bearish_bias = close[i] < weekly_pivot_aligned[i]
        
        # Donchian breakout signals with volume confirmation and weekly bias
        long_breakout = volume_spike and close[i] >= dc_upper_aligned[i] and bullish_bias
        short_breakout = volume_spike and close[i] <= dc_lower_aligned[i] and bearish_bias
        
        # Entry conditions
        long_entry = long_breakout
        short_entry = short_breakout
        
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