#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d volume confirmation and ATR stoploss
# Uses price channel breakouts to capture trends, volume to filter false signals,
# and ATR-based stops to manage risk. Works in bull/bear by following strong moves.
# Target: 50-150 trades over 4 years (12-37/year) to minimize fee drag.

name = "exp_13028_12h_donchian20_1d_volume_atr_v1"
timeframe = "12h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.8
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_donchian(high, low, period):
    """Calculate Donchian channel upper and lower bands"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d volume moving average
    volume_1d = df_1d['volume'].values
    volume_ma_1d = pd.Series(volume_1d).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # Align volume MA to 12h timeframe
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_1d)
    
    # Calculate 12h Donchian channel
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    
    donchian_upper, donchian_lower = calculate_donchian(high_12h, low_12h, DONCHIAN_PERIOD)
    
    # Calculate 12h ATR
    atr_12h = calculate_atr(high_12h, low_12h, close_12h, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if indicators not available
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(volume_ma_aligned[i]) or np.isnan(atr_12h[i]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Check stoploss
        if position == 1:  # long position
            if close_12h[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close_12h[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Volume confirmation (using 1d volume MA aligned to 12h)
        volume_ok = volume_12h[i] > (volume_ma_aligned[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma_aligned[i]) else False
        
        # Breakout above upper band or breakdown below lower band
        breakout_long = volume_ok and close_12h[i] >= donchian_upper[i]
        breakout_short = volume_ok and close_12h[i] <= donchian_lower[i]
        
        # Generate signals
        if position == 0:
            if breakout_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close_12h[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr_12h[i])
            elif breakout_short:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close_12h[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr_12h[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals