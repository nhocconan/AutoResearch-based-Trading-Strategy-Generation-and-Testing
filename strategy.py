#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Donchian breakout with volume confirmation and ATR trailing stop.
# Long when price breaks above 20-period 1d Donchian high with above-average volume.
# Short when price breaks below 20-period 1d Donchian low with above-average volume.
# Uses ATR-based trailing stop to manage risk and lock in profits.
# Designed for 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
# Donchian channels provide clear trend-following structure, volume confirms breakout strength,
# and ATR trailing stop allows profits to run while limiting downside.

name = "exp_13885_12h_donchian1d_vol_trail_v1"
timeframe = "12h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_TRAIL_MULTIPLIER = 2.5

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
    
    # Load 1d data for Donchian channels and ATR
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Donchian channels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    upper_1d, lower_1d = calculate_donchian(high_1d, low_1d, DONCHIAN_PERIOD)
    
    # Align 1d Donchian channels to 12h timeframe
    upper_1d_aligned = align_htf_to_ltf(prices, df_1d, upper_1d)
    lower_1d_aligned = align_htf_to_ltf(prices, df_1d, lower_1d)
    
    # Calculate 1d ATR for trailing stop
    close_1d = df_1d['close'].values
    atr_1d = calculate_atr(high_1d, low_1d, close_1d, ATR_PERIOD)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 12h data for volume confirmation
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume confirmation
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    trail_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VOLUME_MA_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(upper_1d_aligned[i]) or np.isnan(lower_1d_aligned[i]) or np.isnan(atr_1d_aligned[i]) or np.isnan(volume_ma[i]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Update trailing stop
        if position == 1:  # long position
            # Trail stop: max of previous trail or current high minus ATR multiple
            trail_price = max(trail_price, high[i] - (ATR_TRAIL_MULTIPLIER * atr_1d_aligned[i]))
            # Check stop loss
            if close[i] <= trail_price:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # short position
            # Trail stop: min of previous trail or current low plus ATR multiple
            trail_price = min(trail_price, low[i] + (ATR_TRAIL_MULTIPLIER * atr_1d_aligned[i]))
            # Check stop loss
            if close[i] >= trail_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD)
        
        # Donchian breakout signals
        long_signal = volume_ok and close[i] > upper_1d_aligned[i]
        short_signal = volume_ok and close[i] < lower_1d_aligned[i]
        
        # Generate signals
        if position == 0:
            if long_signal:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                trail_price = entry_price - (ATR_TRAIL_MULTIPLIER * atr_1d_aligned[i])
            elif short_signal:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                trail_price = entry_price + (ATR_TRAIL_MULTIPLIER * atr_1d_aligned[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Continue long position
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Continue short position
            signals[i] = -SIGNAL_SIZE
    
    return signals