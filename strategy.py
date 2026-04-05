#!/usr/bin/env python3
"""
Experiment #9593: 4h Donchian(20) breakout + HMA trend + volume confirmation + ATR stoploss.
Hypothesis: Donchian breakouts capture strong trends while HMA(21) filters counter-trend moves. 
Volume confirmation ensures breakout validity. Works in bull (breakouts up) and bear (breakouts down). 
Targets 100-200 total trades over 4 years (25-50/year) to balance opportunity and cost.
Uses 12h HMA for trend filter to avoid overtrading.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_9593_4h_donchian20_hma12h_volume_v1"
timeframe = "4h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
HMA_PERIOD = 21
VOLUME_SPIKE_MULTIPLIER = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_hma(arr, period):
    """Calculate Hull Moving Average"""
    n = int(period)
    if n < 1:
        return arr
    half_n = n // 2
    sqrt_n = int(np.sqrt(n))
    
    # WMA of half period
    arr_pd = pd.Series(arr)
    wma_half = arr_pd.rolling(window=half_n, min_periods=half_n).apply(
        lambda x: np.average(x, weights=np.arange(1, len(x) + 1)), raw=True
    ).values
    
    # WMA of full period
    wma_full = arr_pd.rolling(window=n, min_periods=n).apply(
        lambda x: np.average(x, weights=np.arange(1, len(x) + 1)), raw=True
    ).values
    
    # Raw HMA: 2*WMA(half) - WMA(full)
    raw_hma = 2 * wma_half - wma_full
    
    # Final WMA of sqrt(n)
    raw_series = pd.Series(raw_hma)
    hma = raw_series.rolling(window=sqrt_n, min_periods=sqrt_n).apply(
        lambda x: np.average(x, weights=np.arange(1, len(x) + 1)), raw=True
    ).values
    
    return hma

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
    
    # Load HTF data ONCE before loop (12h for HMA trend filter)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h HMA for trend filter
    close_12h = df_12h['close'].values
    hma_12h = calculate_hma(close_12h, HMA_PERIOD)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # Calculate 4h Donchian channels
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
    start = max(DONCHIAN_PERIOD, HMA_PERIOD, 20) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(hma_12h_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]):
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
        
        # Volume confirmation
        volume_spike = volume[i] > (volume_ma[i] * VOLUME_SPIKE_MULTIPLIER) if not np.isnan(volume_ma[i]) else False
        
        # Breakout conditions with trend filter
        # Long: price breaks above Donchian high AND above 12h HMA (uptrend)
        long_breakout = (close[i] > donchian_high[i]) and (close[i] > hma_12h_aligned[i]) and volume_spike
        # Short: price breaks below Donchian low AND below 12h HMA (downtrend)
        short_breakout = (close[i] < donchian_low[i]) and (close[i] < hma_12h_aligned[i]) and volume_spike
        
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