#!/usr/bin/env python3
"""
Experiment #11271: 6h Camarilla Pivot Reversal with Volume Confirmation
Hypothesis: Camarilla pivot levels (R3/S3 for reversal, R4/S4 for breakout) provide high-probability reversal/continuation signals. 
Volume confirmation ensures institutional participation. Works in both bull and bear markets by adapting to price action at key levels.
Target: 50-150 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_11271_6h_camarilla_pivot_reversal_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
CAMARILLA_PERIOD = 1  # Use previous day's OHLC
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given OHLC"""
    # Camarilla formulas
    pivot = (high + low + close) / 3
    range_val = high - low
    
    r4 = close + range_val * 1.1 / 2
    r3 = close + range_val * 1.1 / 4
    s3 = close - range_val * 1.1 / 4
    s4 = close - range_val * 1.1 / 2
    
    return r4, r3, s3, s4

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate ATR"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    # First TR is just high-low (no previous close)
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate daily Camarilla levels from previous day's OHLC
    # We need to shift the OHLC by 1 to get previous day's levels
    high_prev = df_daily['high'].shift(1).values
    low_prev = df_daily['low'].shift(1).values
    close_prev = df_daily['close'].shift(1).values
    
    # Calculate Camarilla levels for each day
    camarilla_r4 = np.full(len(df_daily), np.nan)
    camarilla_r3 = np.full(len(df_daily), np.nan)
    camarilla_s3 = np.full(len(df_daily), np.nan)
    camarilla_s4 = np.full(len(df_daily), np.nan)
    
    for i in range(len(df_daily)):
        if not np.isnan(high_prev[i]) and not np.isnan(low_prev[i]) and not np.isnan(close_prev[i]):
            r4, r3, s3, s4 = calculate_camarilla(high_prev[i], low_prev[i], close_prev[i])
            camarilla_r4[i] = r4
            camarilla_r3[i] = r3
            camarilla_s3[i] = s3
            camarilla_s4[i] = s4
    
    # Align Camarilla levels to 6h timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_daily, camarilla_r4)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_daily, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_daily, camarilla_s3)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_daily, camarilla_s4)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if Camarilla levels not available
        if np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]):
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
        
        # Camarilla conditions
        # Reversal at S3/R3: price touches S3/R3 and reverses
        # Breakout continuation at S4/R4: price breaks S4/R4 with volume
        
        # Long conditions:
        # 1. Reversal: price touches S3 and closes above it (bounce)
        # 2. Breakout: price breaks above R4 with volume
        long_reversal = (low[i] <= camarilla_s3_aligned[i] and close[i] > camarilla_s3_aligned[i])
        long_breakout = (high[i] > camarilla_r4_aligned[i] and volume_ok)
        
        # Short conditions:
        # 1. Reversal: price touches R3 and closes below it (rejection)
        # 2. Breakout: price breaks below S4 with volume
        short_reversal = (high[i] >= camarilla_r3_aligned[i] and close[i] < camarilla_r3_aligned[i])
        short_breakout = (low[i] < camarilla_s4_aligned[i] and volume_ok)
        
        long_entry = long_reversal or long_breakout
        short_entry = short_reversal or short_breakout
        
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