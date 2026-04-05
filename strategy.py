#!/usr/bin/env python3
"""
Experiment #9540: 4h Donchian(20) Breakout + HMA Trend + Volume Confirmation + ATR Stoploss
Hypothesis: Donchian breakouts capture momentum while HMA filters trend direction and volume confirms strength.
Works in bull (breakouts above upper band) and bear (breakdowns below lower band) with volume confirmation.
Targets 75-200 total trades over 4 years (19-50/year) to balance opportunity and cost.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_9540_4h_donchian20_hma_trend_volume_v1"
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
    
    # WMA function
    def wma(x, window):
        if len(x) < window:
            return np.full_like(x, np.nan)
        weights = np.arange(1, window + 1)
        return np.convolve(x, weights, mode='valid') / (window * (window + 1) / 2)
    
    # Pad array for convolution
    def pad_and_wma(x, window):
        if len(x) < window:
            return np.full_like(x, np.nan)
        return wma(x, window)
    
    # Calculate WMAs
    wma_full = pad_and_wma(arr, n)
    wma_half = pad_and_wma(arr, half_n)
    
    # Handle NaN values
    raw_hma = 2 * wma_half - wma_full
    return pad_and_wma(raw_hma, sqrt_n)

def calculate_donchian_channels(high, low, period):
    """Calculate Donchian channels"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_atr(high, low, close, period):
    """Calculate ATR"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for trend filter)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA for trend filter
    close_1d = df_1d['close'].values
    hma_1d = calculate_hma(close_1d, HMA_PERIOD)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate LTF indicators (4h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels
    upper, lower = calculate_donchian_channels(high, low, DONCHIAN_PERIOD)
    
    # Volume moving average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, 20) + 1
    
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
        breakout_up = close[i] > upper[i]
        breakout_down = close[i] < lower[i]
        
        # Trend filter from 1d HMA
        # For long: price above 1d HMA (uptrend)
        # For short: price below 1d HMA (downtrend)
        uptrend = close[i] > hma_1d_aligned[i]
        downtrend = close[i] < hma_1d_aligned[i]
        
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