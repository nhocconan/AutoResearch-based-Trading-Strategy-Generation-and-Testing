#!/usr/bin/env python3
"""
Experiment #9537: 4h Donchian(20) breakout + HMA(21) trend + volume confirmation.
Hypothesis: Donchian breakouts with trend alignment and volume confirmation capture 
strong momentum moves while avoiding false breakouts. HMA filter ensures we only 
trade in direction of trend. Works in bull (breakouts above) and bear (breakdowns 
below) with trend filter preventing counter-trend trades. Targets 75-200 total 
trades over 4 years (19-50/year) to balance opportunity and cost.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_9537_4h_donchian20_hma_volume_v1"
timeframe = "4h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
HMA_PERIOD = 21
VOLUME_SPIKE_MULTIPLIER = 1.5
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
SIGNAL_SIZE = 0.25

def calculate_hma(close, period):
    """Calculate Hull Moving Average"""
    n = int(period)
    if n < 1:
        return close
    half_n = n // 2
    sqrt_n = int(np.sqrt(n))
    
    # WMA with period n
    def wma(values, window):
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, 'valid') / weights.sum()
    
    # Pad array for convolution
    def padded_wma(arr, window):
        if len(arr) < window:
            return np.full(len(arr), np.nan)
        return np.convolve(arr, np.arange(1, window + 1), 'valid') / np.arange(1, window + 1).sum()
    
    # Calculate WMA(2n) and WMA(n)
    wma_half = np.array([padded_wma(close[:i+1], half_n)[-1] if i+1 >= half_n else np.nan for i in range(len(close))])
    wma_full = np.array([padded_wma(close[:i+1], n)[-1] if i+1 >= n else np.nan for i in range(len(close))])
    
    # Calculate 2*WMA(n/2) - WMA(n)
    raw_hma = 2 * wma_half - wma_full
    
    # WMA(sqrt(n)) of the above
    hma = np.array([padded_wma(raw_hma[:i+1], sqrt_n)[-1] if i+1 >= sqrt_n else np.nan for i in range(len(raw_hma))])
    return hma

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing (equivalent to RMA)"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    
    # Use exponential moving average with alpha = 1/period
    atr = np.full_like(tr, np.nan, dtype=np.float64)
    if len(tr) > 0:
        # First value is simple average
        atr[period-1] = np.nanmean(tr[:period])
        # Subsequent values: ATR = (prev_atr * (period-1) + tr) / period
        for i in range(period, len(tr)):
            if not np.isnan(tr[i]) and not np.isnan(atr[i-1]):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    return atr

def calculate_donchian_channels(high, low, period):
    """Calculate Donchian channels"""
    upper = np.full_like(high, np.nan, dtype=np.float64)
    lower = np.full_like(low, np.nan, dtype=np.float64)
    
    for i in range(len(high)):
        if i >= period - 1:
            upper[i] = np.max(high[i-period+1:i+1])
            lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for trend filter)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA for trend filter
    close_1d = df_1d['close'].values
    hma_1d = calculate_hma(close_1d, HMA_PERIOD)
    
    # Align 1d HMA to 4h timeframe
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate LTF indicators (4h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels
    donchian_upper, donchian_lower = calculate_donchian_channels(high, low, DONCHIAN_PERIOD)
    
    # Volume moving average for spike detection
    volume_ma = np.full_like(volume, np.nan, dtype=np.float64)
    for i in range(len(volume)):
        if i >= 19:  # 20-period MA
            volume_ma[i] = np.mean(volume[i-19:i+1])
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, HMA_PERIOD, ATR_PERIOD, 20) + 1
    
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
        volume_spike = (not np.isnan(volume_ma[i]) and 
                       volume[i] > (volume_ma[i] * VOLUME_SPIKE_MULTIPLIER))
        
        # Trend filter: HMA slope
        hma_rising = (i >= 1 and not np.isnan(hma_1d_aligned[i]) and 
                     not np.isnan(hma_1d_aligned[i-1]) and
                     hma_1d_aligned[i] > hma_1d_aligned[i-1])
        hma_falling = (i >= 1 and not np.isnan(hma_1d_aligned[i]) and 
                      not np.isnan(hma_1d_aligned[i-1]) and
                      hma_1d_aligned[i] < hma_1d_aligned[i-1])
        
        # Breakout conditions
        bullish_breakout = (not np.isnan(donchian_upper[i]) and 
                           close[i] > donchian_upper[i])
        bearish_breakout = (not np.isnan(donchian_lower[i]) and 
                           close[i] < donchian_lower[i])
        
        # Entry conditions
        long_entry = bullish_breakout and hma_rising and volume_spike
        short_entry = bearish_breakout and hma_falling and volume_spike
        
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