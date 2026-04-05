#!/usr/bin/env python3
"""
Experiment #9773: 4h Donchian(20) breakout + 12h HMA trend + volume confirmation + ATR stoploss.
Hypothesis: Donchian breakouts capture strong momentum in trending markets, while 12h HMA filters for trend direction.
Volume confirmation ensures breakouts have conviction. Works in bull (breakouts up) and bear (breakouts down).
Targets 75-200 total trades over 4 years (19-50/year) to balance opportunity and cost.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_9773_4h_donchian_12h_hma_vol_sl_v1"
timeframe = "4h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
HMA_PERIOD = 21
VOLUME_SPIKE_MULTIPLIER = 2.0
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_hma(arr, period):
    """Calculate Hull Moving Average"""
    n = len(arr)
    if n < period:
        return np.full(n, np.nan)
    half = period // 2
    sqrt = int(np.sqrt(period))
    wma2 = np.full(n, np.nan)
    wma1 = np.full(n, np.nan)
    for i in range(half, n):
        wma2[i] = np.nansum(arr[i-half+1:i+1] * np.arange(1, half+1)) / (half * (half + 1) / 2)
    for i in range(period, n):
        wma1[i] = np.nansum(arr[i-period+1:i+1] * np.arange(1, period+1)) / (period * (period + 1) / 2)
    hma = np.full(n, np.nan)
    for i in range(sqrt, n):
        hma[i] = 2 * wma2[i] - wma1[i]
    for i in range(sqrt, n):
        hma[i] = np.nansum(hma[i-sqrt+1:i+1] * np.arange(1, sqrt+1)) / (sqrt * (sqrt + 1) / 2)
    return hma

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = np.full_like(tr, np.nan, dtype=np.float64)
    if len(tr) >= period:
        atr[period-1] = np.nanmean(tr[:period])
        for i in range(period, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (12h for HMA trend)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h HMA
    hma_12h = calculate_hma(df_12h['close'].values, HMA_PERIOD)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # Calculate Donchian channels (4h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian upper/lower
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    for i in range(DONCHIAN_PERIOD-1, n):
        donch_high[i] = np.max(high[i-DONCHIAN_PERIOD+1:i+1])
        donch_low[i] = np.min(low[i-DONCHIAN_PERIOD+1:i+1])
    
    # Volume moving average
    volume_ma = np.full(n, np.nan)
    for i in range(19, n):  # 20-period MA
        volume_ma[i] = np.mean(volume[i-19:i+1])
    
    # ATR
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    start = max(DONCHIAN_PERIOD, HMA_PERIOD, 20) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(hma_12h_aligned[i]):
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
        
        # Trend filter: 12h HMA direction
        uptrend = hma_12h_aligned[i] > hma_12h_aligned[i-1] if i > 0 else False
        downtrend = hma_12h_aligned[i] < hma_12h_aligned[i-1] if i > 0 else False
        
        # Breakout conditions
        breakout_up = close[i] > donch_high[i-1] if i > 0 and not np.isnan(donch_high[i-1]) else False
        breakout_down = close[i] < donch_low[i-1] if i > 0 and not np.isnan(donch_low[i-1]) else False
        
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