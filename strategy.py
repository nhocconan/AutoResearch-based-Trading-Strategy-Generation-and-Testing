#!/usr/bin/env python3
"""
Experiment #9617: 4h Donchian Breakout + HMA Trend + Volume Confirmation.
Hypothesis: Donchian(20) breakouts in the direction of 1d HMA trend with volume confirmation provide high-probability trend continuation signals. Works in bull (breakouts above upper band) and bear (breakdowns below lower band) by following the 1d trend filter. Targets 75-200 total trades over 4 years (19-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_9617_4h_donchian_breakout_hma_trend_v1"
timeframe = "4h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
HMA_PERIOD = 21
VOLUME_SPIKE_MULTIPLIER = 1.5
VOLUME_MA_PERIOD = 20
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def hull_moving_average(arr, period):
    """Calculate Hull Moving Average"""
    n = len(arr)
    if n < period:
        return np.full(n, np.nan)
    
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    # WMA function
    def wma(values, window):
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, 'valid') / weights.sum()
    
    # Calculate WMAs
    wma_half = np.array([np.nan] * (half_period - 1) + [wma(arr[i - half_period + 1:i + 1], half_period) for i in range(half_period - 1, n)])
    wma_full = np.array([np.nan] * (period - 1) + [wma(arr[i - period + 1:i + 1], period) for i in range(period - 1, n)])
    
    # Hull MA: 2*WMA(half) - WMA(full)
    hull_raw = 2 * wma_half - wma_full
    
    # Final WMA of sqrt period
    hma = np.array([np.nan] * (sqrt_period - 1) + [wma(hull_raw[i - sqrt_period + 1:i + 1], sqrt_period) for i in range(sqrt_period - 1, n)])
    
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
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for HMA trend filter)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA
    close_1d = df_1d['close'].values
    hma_1d = hull_moving_average(close_1d, HMA_PERIOD)
    
    # Align 1d HMA to 4h timeframe
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate LTF indicators (4h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels
    highest_high = np.full_like(high, np.nan)
    lowest_low = np.full_like(low, np.nan)
    for i in range(DONCHIAN_PERIOD - 1, n):
        highest_high[i] = np.max(high[i - DONCHIAN_PERIOD + 1:i + 1])
        lowest_low[i] = np.min(low[i - DONCHIAN_PERIOD + 1:i + 1])
    
    # Volume moving average for spike detection
    volume_ma = np.full_like(volume, np.nan, dtype=np.float64)
    for i in range(VOLUME_MA_PERIOD - 1, n):
        volume_ma[i] = np.mean(volume[i - VOLUME_MA_PERIOD + 1:i + 1])
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HMA not available
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
        breakout_up = high[i] >= highest_high[i] if not np.isnan(highest_high[i]) else False
        breakout_down = low[i] <= lowest_low[i] if not np.isnan(lowest_low[i]) else False
        
        # Trend filter: HMA direction
        hma_rising = hma_1d_aligned[i] > hma_1d_aligned[i - 1] if i > 0 and not np.isnan(hma_1d_aligned[i - 1]) else False
        hma_falling = hma_1d_aligned[i] < hma_1d_aligned[i - 1] if i > 0 and not np.isnan(hma_1d_aligned[i - 1]) else False
        
        # Entry conditions
        long_entry = breakout_up and volume_spike and hma_rising
        short_entry = breakout_down and volume_spike and hma_falling
        
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