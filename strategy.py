#!/usr/bin/env python3
"""
Experiment #9709: 4h Donchian Breakout + HMA Trend + Volume Confirmation + ATR Stop.
Hypothesis: Donchian channel breakouts with trend confirmation (HMA) and volume filters
capture strong momentum moves while avoiding whipsaws. Works in bull markets via 
breakouts and bear markets via breakdowns. Targets 75-200 total trades over 4 years
(19-50/year) to balance opportunity and cost. Uses 1d/1w HTF for regime filtering.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_9709_4h_donchian_breakout_hma_vol_v1"
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
    
    half = period // 2
    sqrt = int(np.sqrt(period))
    
    # WMA function
    def wma(values, window):
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, 'valid') / weights.sum()
    
    # Calculate WMAs
    wma_half = np.array([np.nan] * n)
    wma_full = np.array([np.nan] * n)
    
    for i in range(half - 1, n):
        wma_half[i] = wma(arr[i - half + 1:i + 1], half)
    for i in range(period - 1, n):
        wma_full[i] = wma(arr[i - period + 1:i + 1], period)
    
    # Calculate 2*WMA(half) - WMA(full)
    raw_hma = 2 * wma_half - wma_full
    
    # Final WMA of sqrt period
    hma = np.array([np.nan] * n)
    for i in range(sqrt - 1, n):
        hma[i] = wma(raw_hma[i - sqrt + 1:i + 1], sqrt)
    
    return hma

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    
    atr = np.full_like(tr, np.nan, dtype=np.float64)
    if len(tr) >= period:
        atr[period - 1] = np.mean(tr[:period])
        for i in range(period, len(tr)):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for regime filter)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d trend filter (HMA)
    close_1d = df_1d['close'].values
    hma_1d = hull_moving_average(close_1d, HMA_PERIOD)
    hma_1d_prev = np.roll(hma_1d, 1)  # Previous day's HMA for trend
    hma_1d_prev[0] = np.nan
    
    # Align 1d HMA to 4h timeframe
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_prev)
    
    # Calculate LTF indicators (4h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels
    highest = np.full_like(high, np.nan)
    lowest = np.full_like(low, np.nan)
    for i in range(DONCHIAN_PERIOD - 1, n):
        highest[i] = np.max(high[i - DONCHIAN_PERIOD + 1:i + 1])
        lowest[i] = np.min(low[i - DONCHIAN_PERIOD + 1:i + 1])
    
    # Volume moving average for spike detection
    volume_ma = np.full_like(volume, np.nan)
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
        # Skip if HTF data not available
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
        
        # Trend filter: HMA slope (using previous value)
        hma_now = hma_1d_aligned[i]
        hma_prev = hma_1d_aligned[i - 1] if i > 0 else np.nan
        hma_slope_up = hma_now > hma_prev if not np.isnan(hma_now) and not np.isnan(hma_prev) else False
        hma_slope_down = hma_now < hma_prev if not np.isnan(hma_now) and not np.isnan(hma_prev) else False
        
        # Breakout conditions
        breakout_up = high[i] >= highest[i] if not np.isnan(highest[i]) else False
        breakout_down = low[i] <= lowest[i] if not np.isnan(lowest[i]) else False
        
        # Entry conditions
        long_entry = breakout_up and volume_spike and hma_slope_up
        short_entry = breakout_down and volume_spike and hma_slope_down
        
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