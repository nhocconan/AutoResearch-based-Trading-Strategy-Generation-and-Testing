#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Williams Alligator with 12-hour volume confirmation.
# The Alligator (Jaws/Teeth/Lips) identifies trend presence and direction.
# During strong trends, lines are separated and aligned; during consolidation, they intertwine.
# Using 12h volume filter ensures we only trade when institutional participation confirms the trend.
# Works in both bull/bear markets by capturing directional moves with volume validation.
# Target: 100-200 total trades over 4 years (25-50/year) to balance opportunity and cost.

name = "alligator_6h_12h_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
JAWS_PERIOD = 13   # Blue line
TEETH_PERIOD = 8   # Red line
LIPS_PERIOD = 5    # Green line
JAWS_OFFSET = 8
TEETH_OFFSET = 5
LIPS_OFFSET = 3
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_smma(data, period):
    """Calculate Smoothed Moving Average (SMMA)"""
    return pd.Series(data).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values

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
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate Alligator components (SMMA with offsets)
    close_12h = df_12h['close'].values
    jaws_raw = calculate_smma(close_12h, JAWS_PERIOD)
    teeth_raw = calculate_smma(close_12h, TEETH_PERIOD)
    lips_raw = calculate_smma(close_12h, LIPS_PERIOD)
    
    # Apply offsets (shift right by offset periods)
    jaws = np.roll(jaws_raw, JAWS_OFFSET)
    teeth = np.roll(teeth_raw, TEETH_OFFSET)
    lips = np.roll(lips_raw, LIPS_OFFSET)
    
    # Set NaN for invalid periods due to offset and calculation
    jaws[:JAWS_OFFSET + JAWS_PERIOD - 1] = np.nan
    teeth[:TEETH_OFFSET + TEETH_PERIOD - 1] = np.nan
    lips[:LIPS_OFFSET + LIPS_PERIOD - 1] = np.nan
    
    # Align to 6h timeframe
    jaws_aligned = align_htf_to_ltf(prices, df_12h, jaws)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
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
    start = max(
        JAWS_OFFSET + JAWS_PERIOD,
        TEETH_OFFSET + TEETH_PERIOD,
        LIPS_OFFSET + LIPS_PERIOD,
        VOLUME_MA_PERIOD,
        ATR_PERIOD
    ) + 1
    
    for i in range(start, n):
        # Skip if Alligator not ready
        if np.isnan(jaws_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]):
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
        
        # Alligator signals: aligned separation indicates trend
        # Lips > Teeth > Jaws = uptrend (green > red > blue)
        # Lips < Teeth < Jaws = downtrend (green < red < blue)
        uptrend = lips_aligned[i] > teeth_aligned[i] > jaws_aligned[i]
        downtrend = lips_aligned[i] < teeth_aligned[i] < jaws_aligned[i]
        
        # Generate signals
        if position == 0:
            if volume_ok and uptrend:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif volume_ok and downtrend:
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