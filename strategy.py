#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian(20) breakout for direction and 1d volume confirmation for entry timing.
# Long when price breaks above 4h Donchian upper band with above-average volume (1d volume MA > threshold).
# Short when price breaks below 4h Donchian lower band with above-average volume.
# Uses 1h ATR-based stop loss (2x ATR) to manage risk.
# Designed for 60-150 total trades over 4 years (15-37/year) to minimize fee drag.
# Uses session filter (08-20 UTC) to reduce noise trades.
# Fixed size: 0.20.

name = "exp_13834_1h_4h_donchian20_1d_vol_session_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.20
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
    if n < 50:
        return np.zeros(n)
    
    # Load 4h data for Donchian channels ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h Donchian channels
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    upper_4h, lower_4h = calculate_donchian(high_4h, low_4h, DONCHIAN_PERIOD)
    
    # Align 4h Donchian channels to 1h timeframe
    upper_4h_aligned = align_htf_to_ltf(prices, df_4h, upper_4h)
    lower_4h_aligned = align_htf_to_ltf(prices, df_4h, lower_4h)
    
    # Load 1d data for volume confirmation ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d volume moving average
    volume_1d = df_1d['volume'].values
    volume_ma_1d = pd.Series(volume_1d).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # Align 1d volume MA to 1h timeframe
    volume_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_1d)
    
    # 1h data for ATR and price
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Session filter: 08-20 UTC (pre-compute hour array)
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Skip if required data not available
        if (np.isnan(upper_4h_aligned[i]) or np.isnan(lower_4h_aligned[i]) or 
            np.isnan(volume_ma_1d_aligned[i]) or np.isnan(atr[i])):
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
        
        # Volume confirmation: current 1h volume > 1.5 * 1d volume MA
        volume_ok = volume[i] > (volume_ma_1d_aligned[i] * VOLUME_THRESHOLD)
        
        # Donchian breakout signals (using 4h channels)
        long_signal = volume_ok and close[i] > upper_4h_aligned[i]
        short_signal = volume_ok and close[i] < lower_4h_aligned[i]
        
        # Generate signals
        if position == 0:
            if long_signal:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_signal:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on close below 4h Donchian lower band
            if close[i] < lower_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short on close above 4h Donchian upper band
            if close[i] > upper_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals