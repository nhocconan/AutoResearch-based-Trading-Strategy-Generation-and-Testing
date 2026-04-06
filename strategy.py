#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_12851_6h_1d_vwap_deviation_reversion_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
VWAP_PERIOD = 24  # 6d equivalent on 6h chart
DEVIATION_THRESHOLD = 0.02  # 2% deviation from VWAP
REVERSION_THRESHOLD = 0.005  # 0.5% reversion signal
VOLUME_SPIKE_MULTIPLIER = 2.0
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
MAX_HOLD_BARS = 24  # Max 4 days

def calculate_vwap(high, low, close, volume, period):
    """Calculate VWAP using typical price"""
    typical_price = (high + low + close) / 3.0
    vwap_numerator = typical_price * volume
    vwap_denominator = volume
    
    vwap = pd.Series(vwap_numerator).rolling(window=period, min_periods=period).sum().values / \
           pd.Series(vwap_denominator).rolling(window=period, min_periods=period).sum().values
    return vwap

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
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate daily VWAP
    high_d = df_daily['high'].values
    low_d = df_daily['low'].values
    close_d = df_daily['close'].values
    volume_d = df_daily['volume'].values
    
    vwap_daily = calculate_vwap(high_d, low_d, close_d, volume_d, VWAP_PERIOD)
    
    # Align to 6h timeframe
    vwap_aligned = align_htf_to_ltf(prices, df_daily, vwap_daily)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    volume_ma = pd.Series(volume).rolling(window=VWAP_PERIOD, min_periods=VWAP_PERIOD).mean().values
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = max(VWAP_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if VWAP not available
        if np.isnan(vwap_aligned[i]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
        
        # Time-based exit to prevent overtrading
        if bars_since_entry >= MAX_HOLD_BARS:
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
            continue
        
        # Volume spike confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_SPIKE_MULTIPLIER) if not np.isnan(volume_ma[i]) else False
        
        # Calculate deviation from VWAP
        deviation = (close[i] - vwap_aligned[i]) / vwap_aligned[i]
        
        # Mean reversion signals
        reversion_long = volume_ok and deviation <= -DEVIATION_THRESHOLD
        reversion_short = volume_ok and deviation >= DEVIATION_THRESHOLD
        
        # Exit when price reverts halfway back to VWAP
        exit_long = position == 1 and deviation >= -REVERSION_THRESHOLD
        exit_short = position == -1 and deviation <= REVERSION_THRESHOLD
        
        # Generate signals
        if position == 0:
            if reversion_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
                bars_since_entry = 0
            elif reversion_short:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        elif position == 1:
            if exit_long:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            if exit_short:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals