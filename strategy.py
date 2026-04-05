#!/usr/bin/env python3
"""
Experiment #9789: 4h Donchian(20) breakout + HMA trend + volume confirmation + ATR stoploss.
Hypothesis: Donchian breakouts capture trend continuation, HMA(21) filters direction, 
volume confirms conviction, ATR stops manage risk. Works in bull (breakouts up) and 
bear (breakouts down) with trend filter preventing counter-trend entries.
Targets 75-200 total trades over 4 years (19-50/year) to balance opportunity and cost.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_9789_4h_donchian_breakout_hma_volume_v1"
timeframe = "4h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
HMA_PERIOD = 21
VOLUME_SPIKE_MULTIPLIER = 1.5
VOLUME_MA_PERIOD = 20
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
SIGNAL_SIZE = 0.25

def hull_moving_average(arr, period):
    """Calculate Hull Moving Average"""
    n = len(arr)
    if n < period:
        return np.full(n, np.nan)
    
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    # WMA of half period
    wma_half = np.full(n, np.nan)
    for i in range(half_period - 1, n):
        weights = np.arange(1, half_period + 1)
        wma_half[i] = np.sum(arr[i - half_period + 1:i + 1] * weights) / weights.sum()
    
    # WMA of full period
    wma_full = np.full(n, np.nan)
    for i in range(period - 1, n):
        weights = np.arange(1, period + 1)
        wma_full[i] = np.sum(arr[i - period + 1:i + 1] * weights) / weights.sum()
    
    # Raw HMA: 2*WMA(half) - WMA(full)
    hma_raw = 2 * wma_half - wma_full
    
    # Final WMA of sqrt period
    hma = np.full(n, np.nan)
    for i in range(sqrt_period - 1, n):
        weights = np.arange(1, sqrt_period + 1)
        hma[i] = np.sum(hma_raw[i - sqrt_period + 1:i + 1] * weights) / weights.sum()
    
    return hma

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = np.full_like(tr, np.nan, dtype=np.float64)
    
    if len(tr) >= period:
        atr[period - 1] = np.nanmean(tr[:period])
        for i in range(period, len(tr)):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for trend filter)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA for trend filter
    hma_1d = hull_moving_average(df_1d['close'].values, HMA_PERIOD)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate LTF indicators (4h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(DONCHIAN_PERIOD - 1, n):
        donchian_high[i] = np.max(high[i - DONCHIAN_PERIOD + 1:i + 1])
        donchian_low[i] = np.min(low[i - DONCHIAN_PERIOD + 1:i + 1])
    
    # Volume moving average for spike detection
    volume_ma = np.full(n, np.nan)
    for i in range(VOLUME_MA_PERIOD - 1, n):
        volume_ma[i] = np.mean(volume[i - VOLUME_MA_PERIOD + 1:i + 1])
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, HMA_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HMA data not available
        if np.isnan(hma_1d_aligned[i]):
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
        
        # Breakout conditions
        breakout_up = high[i] >= donchian_high[i]  # Using high to catch breakout
        breakout_down = low[i] <= donchian_low[i]   # Using low to catch breakdown
        
        # Trend filter: HMA slope (simplified: current vs previous)
        hma_now = hma_1d_aligned[i]
        hma_prev = hma_1d_aligned[i - 1] if i > 0 else hma_now
        uptrend = hma_now > hma_prev
        downtrend = hma_now < hma_prev
        
        # Entry conditions
        long_entry = breakout_up and volume_spike and uptrend
        short_entry = breakout_down and volume_spike and downtrend
        
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