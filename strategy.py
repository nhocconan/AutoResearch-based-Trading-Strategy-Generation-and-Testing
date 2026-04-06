#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1-week Donchian(20) breakout with 1-day volume confirmation and 1-day ATR stop loss.
# Goes long when price breaks above 1w Donchian upper band with above-average daily volume, short when breaks below.
# Uses 1d ATR-based stop loss to manage risk. Designed for 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
# Multi-timeframe approach: weekly structure for trend, daily for volume confirmation, 12h for entry timing.

name = "exp_13848_12h_1w_donchian20_1d_vol_atr_v1"
timeframe = "12h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

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
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1w data for Donchian channels ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w Donchian channels
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    upper_1w, lower_1w = calculate_donchian(high_1w, low_1w, DONCHIAN_PERIOD)
    
    # Align 1w Donchian to 12h timeframe
    upper_1w_aligned = align_htf_to_ltf(prices, df_1w, upper_1w)
    lower_1w_aligned = align_htf_to_ltf(prices, df_1w, lower_1w)
    
    # Load 1d data for volume and ATR ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d volume moving average
    volume_1d = df_1d['volume'].values
    volume_ma_1d = pd.Series(volume_1d).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    volume_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_1d)
    
    # Calculate 1d ATR for stop loss
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    atr_1d = calculate_atr(high_1d, low_1d, close_1d, ATR_PERIOD)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 12h data for price and volume
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(upper_1w_aligned[i]) or np.isnan(lower_1w_aligned[i]) or np.isnan(volume_ma_1d_aligned[i]) or np.isnan(atr_1d_aligned[i]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Check stops
        if position == 1:  # long position
            # Check stop loss
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # short position
            # Check stop loss
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Volume confirmation (using 1d average volume)
        volume_ok = volume[i] > (volume_ma_1d_aligned[i] * VOLUME_THRESHOLD)
        
        # Donchian breakout signals (using 1w channels)
        long_signal = volume_ok and close[i] > upper_1w_aligned[i]
        short_signal = volume_ok and close[i] < lower_1w_aligned[i]
        
        # Generate signals
        if position == 0:
            if long_signal:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr_1d_aligned[i])
            elif short_signal:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr_1d_aligned[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on close below 1w Donchian lower band
            if close[i] < lower_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short on close above 1w Donchian upper band
            if close[i] > upper_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals