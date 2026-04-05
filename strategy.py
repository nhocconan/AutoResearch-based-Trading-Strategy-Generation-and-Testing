#!/usr/bin/env python3
"""
Experiment #9633: 4h Donchian(20) breakout + 12h HMA trend + volume confirmation + ATR stoploss.
Hypothesis: Donchian channel breakouts capture strong momentum moves, filtered by 12h HMA trend
to avoid counter-trend trades, with volume confirmation to ensure breakout validity.
ATR-based stoploss manages risk. Designed to work in both bull (breakouts up) and bear
(breakouts down) markets with controlled trade frequency (target: 75-200 total trades over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_9633_4h_donchian_12hma_volume_v1"
timeframe = "4h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
HMA_PERIOD = 21
VOLUME_SPIKE_MULTIPLIER = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_hma(arr, period):
    """Calculate Hull Moving Average"""
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    # WMA function
    def wma(values, window):
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, 'valid') / weights.sum()
    
    # Handle edge cases with padding
    n = len(arr)
    if n < period:
        return np.full(n, np.nan)
    
    wma_half = np.full(n, np.nan)
    wma_full = np.full(n, np.nan)
    
    for i in range(half_period - 1, n):
        wma_half[i] = wma(arr[i - half_period + 1:i + 1], half_period)
    for i in range(period - 1, n):
        wma_full[i] = wma(arr[i - period + 1:i + 1], period)
    
    wma_2half = 2 * wma_half
    diff = wma_2half - wma_full
    
    hma = np.full(n, np.nan)
    for i in range(sqrt_period - 1, n):
        hma[i] = wma(diff[i - sqrt_period + 1:i + 1], sqrt_period)
    
    return hma

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
    tr[0] = tr1[0]  # First value
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (12h for HMA trend)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h HMA
    close_12h = df_12h['close'].values
    hma_12h = calculate_hma(close_12h, HMA_PERIOD)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # Calculate 4h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels
    upper, lower = calculate_donchian(high, low, DONCHIAN_PERIOD)
    
    # Volume moving average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, HMA_PERIOD * 2, 20) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(hma_12h_aligned[i]) or np.isnan(upper[i]) or np.isnan(lower[i]):
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
        
        # Trend filter: HMA direction (rising = bullish, falling = bearish)
        hma_rising = hma_12h_aligned[i] > hma_12h_aligned[i-1] if i > 0 else False
        hma_falling = hma_12h_aligned[i] < hma_12h_aligned[i-1] if i > 0 else False
        
        # Breakout signals with trend filter and volume confirmation
        long_breakout = (close[i] >= upper[i]) and hma_rising and volume_spike
        short_breakout = (close[i] <= lower[i]) and hma_falling and volume_spike
        
        # Entry conditions
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
</|reserved_token_163247|>