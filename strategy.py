#!/usr/bin/env python3
"""
Experiment #9623: 4h Donchian Breakout + Volume Confirmation + Regime Filter.
Hypothesis: Donchian(20) breakouts aligned with 12h trend (HMA) and volume spikes 
provide high-probability trend-following entries. Uses 1d ATR for stoploss. 
Designed for 75-200 total trades over 4 years (19-50/year) to minimize fee drag 
while maintaining statistical validity. Works in bull (breakouts up) and bear 
(breakouts down) with regime filter avoiding whipsaws in ranging markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_9623_4h_donchian_breakout_volume_regime_v1"
timeframe = "4h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOLUME_SPIKE_MULTIPLIER = 1.5
ADX_PERIOD = 14
ADX_THRESHOLD = 20  # Lower threshold for more frequent trends
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_donchian_channels(high, low, period):
    """Calculate Donchian channels"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_hma(arr, period):
    """Calculate Hull Moving Average"""
    n = len(arr)
    half = period // 2
    sqrt = int(np.sqrt(period))
    
    if half < 1:
        half = 1
    if sqrt < 1:
        sqrt = 1
        
    # WMA function
    def wma(values, window):
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, 'valid') / weights.sum()
    
    # Calculate WMAs
    wma_half = np.array([wma(arr[i-half+1:i+1], half) if i >= half-1 else np.nan for i in range(n)])
    wma_full = np.array([wma(arr[i-period+1:i+1], period) if i >= period-1 else np.nan for i in range(n)])
    wma_sqrt = np.array([wma(arr[i-sqrt+1:i+1], sqrt) if i >= sqrt-1 else np.nan for i in range(n)])
    
    # HMA = 2*WMA(half) - WMA(full)
    hma_raw = 2 * wma_half - wma_full
    # Final WMA
    hma = np.array([wma(hma_raw[max(0, i-sqrt+1):i+1], sqrt) if i >= sqrt-1 and not np.isnan(hma_raw[i]) else np.nan for i in range(n)])
    
    return hma

def calculate_adx(high, low, close, period):
    """Calculate ADX using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    
    plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    
    # Handle first element
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    # Smooth using Wilder's smoothing (alpha = 1/period)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values / (atr + 1e-10)
    
    dx = np.where((plus_di + minus_di) != 0, 
                  100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return adx

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    
    # Handle first element
    tr[0] = tr1[0]
    
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (12h for trend, 1d for ATR)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 12h HMA for trend filter
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    hma_12h = calculate_hma(close_12h, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # Calculate 1d ATR for stoploss (using 1d data)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    atr_1d = calculate_atr(high_1d, low_1d, close_1d, ATR_PERIOD)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate LTF indicators (4h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels
    donch_upper, donch_lower = calculate_donchian_channels(high, low, DONCHIAN_PERIOD)
    
    # Volume moving average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ADX for regime filtering
    adx = calculate_adx(high, low, close, ADX_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, 20, ADX_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(hma_12h_aligned[i]) or np.isnan(atr_1d_aligned[i]):
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
        
        # Regime filter: ADX > 20 for trending market
        trending = adx[i] > ADX_THRESHOLD
        
        # Donchian breakout conditions
        breakout_up = close[i] > donch_upper[i]
        breakout_down = close[i] < donch_lower[i]
        
        # Entry conditions: breakout + volume + trend alignment
        long_entry = breakout_up and volume_spike and trending and hma_12h_aligned[i] > hma_12h_aligned[i-1]
        short_entry = breakout_down and volume_spike and trending and hma_12h_aligned[i] < hma_12h_aligned[i-1]
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr_1d_aligned[i])
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr_1d_aligned[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals