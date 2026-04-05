#!/usr/bin/env python3
"""
Experiment #9791: 6h Donchian Breakout + Weekly Pivot Direction + Volume Confirmation.
Hypothesis: Donchian(20) breakouts combined with weekly pivot direction (from weekly high/low) 
and volume confirmation provide high-probability trend continuation signals. 
Weekly pivot direction filters breakouts to trade only in the direction of the weekly trend. 
Works in bull markets (breakouts above weekly pivot) and bear markets (breakdowns below weekly pivot).
Targets 100-200 total trades over 4 years (25-50/year) to balance opportunity and cost.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf

name = "exp_9791_6h_donchian_breakout_weekly_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOLUME_SPIKE_MULTIPLIER = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
WEEKLY_LOOKBACK = 5  # weeks for pivot calculation

def calculate_donchian_channels(high, low, period):
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

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for weekly pivot calculation)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate weekly high/low from daily data
    # Weekly high = max of high over past 5 trading days
    # Weekly low = min of low over past 5 trading days
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    weekly_high = pd.Series(high_1d).rolling(window=WEEKLY_LOOKBACK, min_periods=WEEKLY_LOOKBACK).max().values
    weekly_low = pd.Series(low_1d).rolling(window=WEEKLY_LOOKBACK, min_periods=WEEKLY_LOOKBACK).min().values
    
    # Align weekly levels to 6h timeframe
    weekly_high_aligned = align_htf_to_ltf(prices, df_1d, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1d, weekly_low)
    
    # Calculate LTF indicators (6h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels
    donchian_upper, donchian_lower = calculate_donchian_channels(high, low, DONCHIAN_PERIOD)
    
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
        # Skip if weekly data not available
        if np.isnan(weekly_high_aligned[i]) or np.isnan(weekly_low_aligned[i]):
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
        
        # Weekly pivot direction: 
        # If price > weekly high -> bullish bias (look for long breakouts)
        # If price < weekly low -> bearish bias (look for short breakdowns)
        # If between weekly high and low -> no clear bias (stay flat or wait)
        price_vs_weekly = close[i] - weekly_high_aligned[i]
        is_bullish_bias = price_vs_weekly > 0  # Price above weekly high
        is_bearish_bias = close[i] < weekly_low_aligned[i]  # Price below weekly low
        
        # Donchian breakout signals with volume confirmation and weekly bias
        long_breakout = (close[i] >= donchian_upper[i]) and volume_spike and is_bullish_bias
        short_breakout = (close[i] <= donchian_lower[i]) and volume_spike and is_bearish_bias
        
        # Generate signals
        if position == 0:
            if long_breakout:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_breakout:
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