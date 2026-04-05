#!/usr/bin/env python3
"""
Experiment #9699: 6h Donchian(20) Breakout + Weekly Pivot Direction + Volume Confirmation.
Hypothesis: Combining daily Donchian breakouts with weekly pivot levels (from 1d OHLC) 
provides high-probability trend continuation signals. Volume confirmation filters false breakouts.
Works in bull markets (breakouts above weekly pivot) and bear markets (breakdowns below weekly pivot).
Targets 75-150 total trades over 4 years (19-38/year) to balance opportunity and cost.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_9699_6h_donchian20_weekly_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
WEEKLY_PIVOT_PERIOD = 5  # 5 trading days = 1 week
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
    Calculate weekly pivot from daily OHLC
    Using 5-day aggregation (Monday-Friday)
    PP = (High + Low + Close) / 3
    R1 = 2*PP - Low
    S1 = 2*PP - High
    R2 = PP + (High - Low)
    S2 = PP - (High - Low)
    R3 = High + 2*(PP - Low)
    S3 = Low - 2*(High - PP)
    """
    # We'll use the weekly high, low, close from the past 5 days
    # For simplicity, using the same formula as daily but on weekly aggregates
    high_weekly = pd.Series(high).rolling(window=WEEKLY_PIVOT_PERIOD, min_periods=WEEKLY_PIVOT_PERIOD).max().values
    low_weekly = pd.Series(low).rolling(window=WEEKLY_PIVOT_PERIOD, min_periods=WEEKLY_PIVOT_PERIOD).min().values
    close_weekly = pd.Series(close).rolling(window=WEEKLY_PIVOT_PERIOD, min_periods=WEEKLY_PIVOT_PERIOD).last().values
    
    pp = (high_weekly + low_weekly + close_weekly) / 3
    r1 = 2 * pp - low_weekly
    s1 = 2 * pp - high_weekly
    r2 = pp + (high_weekly - low_weekly)
    s2 = pp - (high_weekly - low_weekly)
    r3 = high_weekly + 2 * (pp - low_weekly)
    s3 = low_weekly - 2 * (high_weekly - pp)
    
    return pp, r1, r2, r3, s1, s2, s3

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for weekly pivot calculation)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d weekly pivot levels (using 5-day aggregation)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot levels from daily data
    pp_1d, r1_1d, r2_1d, r3_1d, s1_1d, s2_1d, s3_1d = calculate_weekly_pivot(high_1d, low_1d, close_1d)
    
    # Align weekly pivot levels to 6h timeframe
    pp_1d_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # Calculate LTF indicators (6h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    donch_upper, donch_lower = calculate_donchian(high, low, DONCHIAN_PERIOD)
    
    # Volume moving average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, WEEKLY_PIVOT_PERIOD*2, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(pp_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]):
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
        
        # Breakout conditions with weekly pivot filter
        # Long: price breaks above Donchian upper AND above weekly R3 (bullish bias)
        long_breakout = (not np.isnan(donch_upper[i]) and 
                         close[i] > donch_upper[i] and 
                         close[i] > r3_1d_aligned[i] and 
                         volume_spike)
        
        # Short: price breaks below Donchian lower AND below weekly S3 (bearish bias)
        short_breakout = (not np.isnan(donch_lower[i]) and 
                          close[i] < donch_lower[i] and 
                          close[i] < s3_1d_aligned[i] and 
                          volume_spike)
        
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