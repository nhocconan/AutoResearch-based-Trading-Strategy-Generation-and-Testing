#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Williams Alligator (3 SMAs: Jaw=13, Teeth=8, Lips=5) with daily pivot breakout
# and volume confirmation. Alligator lines act as dynamic support/resistance; price crossing
# all three lines with volume indicates strong trend. Daily pivot adds confluence for
# institutional interest zones. Works in bull markets (bullish alignment above pivot) and
# bear markets (bearish alignment below pivot). Target: 50-150 total trades over 4 years.

name = "exp_13292_12h_alligator_pivot_vol_v1"
timeframe = "12h"
leverage = 1.0

# Parameters
JAW_PERIOD = 13   # Alligator Jaw (slowest)
TEETH_PERIOD = 8  # Alligator Teeth
LIPS_PERIOD = 5   # Alligator Lips (fastest)
PIVOT_PERIOD = 1  # Daily pivot
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_sma(arr, period):
    """Calculate Simple Moving Average"""
    return pd.Series(arr).rolling(window=period, min_periods=period).mean().values

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
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Alligator components (12h data)
    close = prices['close'].values
    jaw = calculate_sma(close, JAW_PERIOD)
    teeth = calculate_sma(close, TEETH_PERIOD)
    lips = calculate_sma(close, LIPS_PERIOD)
    
    # Calculate daily pivots
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Standard pivot point: P = (H + L + C) / 3
    # Support 1: S1 = 2*P - H
    # Resistance 1: R1 = 2*P - L
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = 2 * pivot - high_1d
    s1 = 2 * pivot - low_1d
    
    # Align pivots to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # 12h indicators
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(JAW_PERIOD, TEETH_PERIOD, LIPS_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if indicators not ready
        if np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or \
           np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]):
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
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Alligator alignment: Lips > Teeth > Jaw = bullish, Lips < Teeth < Jaw = bearish
        bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
        bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        # Breakout signals with pivot confluence
        breakout_up = volume_ok and bullish_alignment and (high[i] > r1_aligned[i-1])
        breakout_down = volume_ok and bearish_alignment and (low[i] < s1_aligned[i-1])
        
        # Generate signals
        if position == 0:
            if breakout_up:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif breakout_down:
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