#!/usr/bin/env python3
"""
Experiment #9503: 4h Donchian Breakout + HMA Trend + Volume + ATR Stop
Hypothesis: Donchian(20) breakouts with 4h HMA(21) trend filter and volume confirmation
provide robust trend-following signals that work in both bull and bear markets by
only taking breakouts in the direction of the higher timeframe trend. Targets 75-200
trades over 4 years (19-50/year) to minimize fee drag while capturing strong moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_9503_4h_donchian_breakout_hma_volume_v1"
timeframe = "4h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
HMA_PERIOD = 21
VOLUME_SPIKE_MULTIPLIER = 1.5
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
SIGNAL_SIZE = 0.25

def calculate_hma(arr, period):
    """Calculate Hull Moving Average"""
    n = len(arr)
    if n < period:
        return np.full(n, np.nan)
    half = period // 2
    sqrt = int(np.sqrt(period))
    
    # WMA of half period
    wma_half = np.full(n, np.nan)
    for i in range(half - 1, n):
        wma_half[i] = np.nansum(arr[i - half + 1:i + 1] * np.arange(1, half + 1)) / (half * (half + 1) / 2)
    
    # WMA of full period
    wma_full = np.full(n, np.nan)
    for i in range(period - 1, n):
        wma_full[i] = np.nansum(arr[i - period + 1:i + 1] * np.arange(1, period + 1)) / (period * (period + 1) / 2)
    
    # HMA = 2*WMA(half) - WMA(full)
    hma = 2 * wma_half - wma_full
    
    # Final WMA of sqrt period
    hma_final = np.full(n, np.nan)
    for i in range(sqrt - 1, n):
        hma_final[i] = np.nansum(hma[i - sqrt + 1:i + 1] * np.arange(1, sqrt + 1)) / (sqrt * (sqrt + 1) / 2)
    
    return hma_final

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = np.full_like(tr, np.nan, dtype=np.float64)
    
    # Wilder's smoothing: first value is simple average, then smoothed
    if len(tr) >= period:
        atr[period - 1] = np.nanmean(tr[:period])
        for i in range(period, len(tr)):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for trend filter)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate LTF indicators (4h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels
    donchian_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    donchian_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # HMA for trend
    hma = calculate_hma(close, HMA_PERIOD)
    
    # Volume moving average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, HMA_PERIOD, 20, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if indicators not ready
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(hma[i]) or np.isnan(ema_1d_aligned[i]):
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
        breakout_up = close[i] > donchian_high[i]
        breakout_down = close[i] < donchian_low[i]
        
        # Trend filters
        uptrend = hma[i] > ema_1d_aligned[i]  # Price above both HMA and 1d EMA
        downtrend = hma[i] < ema_1d_aligned[i]
        
        # Entry conditions
        long_entry = breakout_up and uptrend and volume_spike
        short_entry = breakout_down and downtrend and volume_spike
        
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