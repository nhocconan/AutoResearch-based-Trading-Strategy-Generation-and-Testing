#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using Camarilla pivot levels from 1d timeframe with volume confirmation.
# Goes long when price breaks above R4 level with above-average volume, short when breaks below S4 level with volume.
# Uses daily pivot context to filter trades: only trade long when price above daily pivot, short when below.
# Camarilla levels provide precise support/resistance, reducing false breakouts. Designed for 50-150 total trades over 4 years.

name = "exp_13847_6h_camarilla1d_vol_pivot_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
CAMARILLA_MULT = 1.1  # Standard Camarilla multiplier
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for the day"""
    pivot = (high + low + close) / 3
    range_val = high - low
    # Camarilla levels
    r4 = close + range_val * CAMARILLA_MULT * 1.5
    r3 = close + range_val * CAMARILLA_MULT * 1.25
    r2 = close + range_val * CAMARILLA_MULT * 1.166
    r1 = close + range_val * CAMARILLA_MULT * 1.083
    s1 = close - range_val * CAMARILLA_MULT * 1.083
    s2 = close - range_val * CAMARILLA_MULT * 1.166
    s3 = close - range_val * CAMARILLA_MULT * 1.25
    s4 = close - range_val * CAMARILLA_MULT * 1.5
    return r4, r3, r2, r1, pivot, s1, s2, s3, s4

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
    
    # Load 1d data for Camarilla levels and pivot filter ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels and pivot for each day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    r4_1d, r3_1d, r2_1d, r1_1d, pivot_1d, s1_1d, s2_1d, s3_1d, s4_1d = calculate_camarilla(high_1d, low_1d, close_1d)
    
    # Align 1d Camarilla levels and pivot to 6h timeframe
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    
    # 6h data for price action, volume, and ATR
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Volume confirmation
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(r4_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or np.isnan(pivot_1d_aligned[i]) or np.isnan(volume_ma[i]):
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
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD)
        
        # Price relative to daily pivot for context filter
        above_pivot = close[i] > pivot_1d_aligned[i]
        below_pivot = close[i] < pivot_1d_aligned[i]
        
        # Camarilla breakout signals with pivot filter
        long_signal = volume_ok and above_pivot and close[i] > r4_1d_aligned[i]
        short_signal = volume_ok and below_pivot and close[i] < s4_1d_aligned[i]
        
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
            # Exit long on close below S3 level (mean reversion expectation)
            if close[i] < s3_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short on close above R3 level (mean reversion expectation)
            if close[i] > r3_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals