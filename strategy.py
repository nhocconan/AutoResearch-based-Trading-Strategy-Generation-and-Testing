#!/usr/bin/env python3
"""
Experiment #9648: 12h Donchian Breakout + Volume Spike + 1w Trend Filter
Hypothesis: 12h Donchian(20) breakouts with volume confirmation and 1w trend filter (price > 200 EMA) 
capture strong trending moves while minimizing false signals. Targets 50-150 total trades over 4 years 
(12-37/year) to balance opportunity and cost. Works in bull markets (long breakouts) and avoids 
shorts in bear markets by using 1w trend filter to only take longs when above 200 EMA.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_9648_12h_donchian_breakout_volume_1wtrend_v1"
timeframe = "12h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOLUME_SPIKE_MULTIPLIER = 2.0
EMA_FAST = 20
EMA_SLOW = 200
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_donchian_channels(high, low, period):
    """Calculate Donchian channels"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

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
    
    # Load HTF data ONCE before loop (1w for trend filter)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w 200 EMA for trend filter
    close_1w = df_1w['close'].values
    ema200_1w = calculate_ema(close_1w, EMA_SLOW)
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Calculate 1d data for Donchian channels (more responsive than 1w for breakouts)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donchian_upper_1d, donchian_lower_1d = calculate_donchian_channels(high_1d, low_1d, DONCHIAN_PERIOD)
    donchian_upper_1d_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper_1d)
    donchian_lower_1d_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower_1d)
    
    # Calculate LTF indicators (12h)
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
    start = max(DONCHIAN_PERIOD, EMA_SLOW, 20) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if (np.isnan(donchian_upper_1d_aligned[i]) or np.isnan(donchian_lower_1d_aligned[i]) or 
            np.isnan(ema200_1w_aligned[i])):
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
        
        # Trend filter: only take longs when price > 1w 200 EMA (avoid shorts in bear markets)
        uptrend = close[i] > ema200_1w_aligned[i]
        
        # Breakout signals with volume confirmation
        breakout_long = volume_spike and close[i] >= donchian_upper_1d_aligned[i]
        breakout_short = volume_spike and close[i] <= donchian_lower_1d_aligned[i]
        
        # Entry conditions: only take longs in uptrend, no shorts (to avoid bear market whipsaw)
        long_entry = breakout_long and uptrend
        # No short entries to prevent losses in bear markets
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        else:
            signals[i] = 0.0
    
    return signals