#!/usr/bin/env python3
"""
Experiment #9739: 6h Donchian Breakout + Weekly Pivot Direction + Volume Confirmation
Hypothesis: Donchian(20) breakouts combined with weekly pivot direction (trend filter) 
and volume confirmation provide high-probability trend-following entries in both bull 
and bear markets. Weekly pivot determines primary trend direction (above/below pivot), 
while Donchian breakouts capture momentum. Volume confirmation reduces false breakouts.
Target: 75-150 total trades over 4 years (19-38/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_9739_6h_donchian_breakout_weekly_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOLUME_SPIKE_MULTIPLIER = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_true_range(high, low, close):
    """Calculate True Range"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    return tr

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr = calculate_true_range(high, low, close)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_weekly_pivot(high, low, close):
    """
    Calculate weekly pivot point and support/resistance levels
    Pivot = (High + Low + Close) / 3
    R1 = 2*Pivot - Low
    S1 = 2*Pivot - High
    R2 = Pivot + (High - Low)
    S2 = Pivot - (High - Low)
    """
    pivot = (high + low + close) / 3.0
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    return pivot, r1, r2, s1, s2

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (weekly for pivot calculation)
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot levels
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    pivot_weekly, r1_weekly, r2_weekly, s1_weekly, s2_weekly = calculate_weekly_pivot(
        high_weekly, low_weekly, close_weekly
    )
    
    # Align weekly levels to 6h timeframe
    pivot_weekly_aligned = align_htf_to_ltf(prices, df_weekly, pivot_weekly)
    r1_weekly_aligned = align_htf_to_ltf(prices, df_weekly, r1_weekly)
    r2_weekly_aligned = align_htf_to_ltf(prices, df_weekly, r2_weekly)
    s1_weekly_aligned = align_htf_to_ltf(prices, df_weekly, s1_weekly)
    s2_weekly_aligned = align_htf_to_ltf(prices, df_weekly, s2_weekly)
    
    # Calculate LTF indicators (6h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels
    donchian_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    donchian_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
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
        if np.isnan(pivot_weekly_aligned[i]):
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
        
        # Trend filter: price above/below weekly pivot
        above_pivot = close[i] > pivot_weekly_aligned[i]
        below_pivot = close[i] < pivot_weekly_aligned[i]
        
        # Donchian breakout conditions
        breakout_high = close[i] > donchian_high[i-1]  # Break above previous period's high
        breakout_low = close[i] < donchian_low[i-1]    # Break below previous period's low
        
        # Entry conditions
        long_entry = breakout_high and volume_spike and above_pivot
        short_entry = breakout_low and volume_spike and below_pivot
        
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