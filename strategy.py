#!/usr/bin/env python3
"""
Experiment #11379: 6h Williams Alligator with 12h Trend and Volume Confirmation
Hypothesis: The Williams Alligator (Jaw/Teeth/Lips) identifies trend phases. 
In 6h timeframe, we use 12h EMA for trend filter and volume confirmation to avoid false signals.
Works in bull markets (teeth above jaw = uptrend) and bear markets (teeth below jaw = downtrend).
Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_11379_6w_alligator_12h_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

# Williams Alligator parameters (default: 13,8,5 with 8,5,3 shifts)
JAW_PERIOD = 13
TEETH_PERIOD = 8
LIPS_PERIOD = 5
JAW_SHIFT = 8
TEETH_SHIFT = 5
LIPS_SHIFT = 3

# Additional filters
TREND_PERIOD = 21  # 12h EMA for trend filter
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_alligator_series(data, period, shift):
    """Calculate Alligator line (SMMA with shift)"""
    # Smoothed Moving Average (SMMA) approximation using EMA with alpha=1/period
    smoothed = pd.Series(data).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    # Apply shift (lookback)
    shifted = np.roll(smoothed, shift)
    # First 'shift' values are invalid
    shifted[:shift] = np.nan
    return shifted

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

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
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA for trend filter
    ema_12h = calculate_ema(df_12h['close'].values, TREND_PERIOD)
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams Alligator lines
    jaw = calculate_alligator_series(close, JAW_PERIOD, JAW_SHIFT)
    teeth = calculate_alligator_series(close, TEETH_PERIOD, TEETH_SHIFT)
    lips = calculate_alligator_series(close, LIPS_PERIOD, LIPS_SHIFT)
    
    # Volume and ATR
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(JAW_PERIOD + JAW_SHIFT, TEETH_PERIOD + TEETH_SHIFT, 
                LIPS_PERIOD + LIPS_SHIFT, TREND_PERIOD, VOLUME_MA_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if 12h EMA not available
        if np.isnan(ema_12h_aligned[i]):
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
        
        # Alligator conditions: teeth above jaw = uptrend, teeth below jaw = downtrend
        # Avoid trading when alligator is sleeping (all lines intertwined)
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        
        # Check if values are valid
        if np.isnan(jaw_val) or np.isnan(teeth_val) or np.isnan(lips_val):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Alligator awake: jaws, teeth, lips are not all intertwined
        # Uptrend: lips > teeth > jaw
        # Downtrend: lips < teeth < jaw
        uptrend_alligator = lips_val > teeth_val and teeth_val > jaw_val
        downtrend_alligator = lips_val < teeth_val and teeth_val < jaw_val
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Trend filter (12h EMA)
        uptrend_12h = close[i] > ema_12h_aligned[i]
        downtrend_12h = close[i] < ema_12h_aligned[i]
        
        # Entry conditions: Alligator direction + 12h trend + volume
        long_entry = uptrend_alligator and uptrend_12h and volume_ok
        short_entry = downtrend_alligator and downtrend_12h and volume_ok
        
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