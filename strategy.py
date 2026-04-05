#!/usr/bin/env python3
"""
exp_7399_6d_pivot3levels_volatility_breakout_v1
Hypothesis: 6-hour volatility breakout using 1-day pivot points (R1/S1, R2/S2, R3/S3).
Breakout above R3 or below S3 with volume confirmation captures strong trending moves.
Uses 1-day pivot levels for robust support/resistance that works in both bull/bear markets.
Target: 50-150 trades over 4 years (12-37/year) with discrete sizing (0.25) to minimize fees.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7399_6d_pivot3levels_volatility_breakout_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
PIVOT_LENGTH = 1  # Use previous day's OHLC for pivot calculation
VOL_MA_PERIOD = 20
VOL_BREAKOUT_THRESHOLD = 2.0
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
MAX_HOLD_BARS = 12  # ~3 days max hold

def calculate_pivot_levels(high, low, close):
    """Calculate classic pivot points and support/resistance levels"""
    pivot = (high + low + close) / 3.0
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    r3 = high + 2 * (pivot - low)
    s3 = low - 2 * (high - pivot)
    return pivot, r1, r2, r3, s1, s2, s3

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1d for pivot levels
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d pivot levels from previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot levels for each day
    pivot_vals = np.full(len(high_1d), np.nan)
    r1_vals = np.full(len(high_1d), np.nan)
    r2_vals = np.full(len(high_1d), np.nan)
    r3_vals = np.full(len(high_1d), np.nan)
    s1_vals = np.full(len(high_1d), np.nan)
    s2_vals = np.full(len(high_1d), np.nan)
    s3_vals = np.full(len(high_1d), np.nan)
    
    for i in range(len(high_1d)):
        if i >= PIVOT_LENGTH:  # Need previous day's data
            pivot, r1, r2, r3, s1, s2, s3 = calculate_pivot_levels(
                high_1d[i-1], low_1d[i-1], close_1d[i-1]
            )
            pivot_vals[i] = pivot
            r1_vals[i] = r1
            r2_vals[i] = r2
            r3_vals[i] = r3
            s1_vals[i] = s1
            s2_vals[i] = s2
            s3_vals[i] = s3
    
    # Align pivot levels to 6h timeframe
    pivot_6h = align_htf_to_ltf(prices, df_1d, pivot_vals)
    r1_6h = align_htf_to_ltf(prices, df_1d, r1_vals)
    r2_6h = align_htf_to_ltf(prices, df_1d, r2_vals)
    r3_6h = align_htf_to_ltf(prices, df_1d, r3_vals)
    s1_6h = align_htf_to_ltf(prices, df_1d, s1_vals)
    s2_6h = align_htf_to_ltf(prices, df_1d, s2_vals)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3_vals)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=VOL_MA_PERIOD, min_periods=VOL_MA_PERIOD).mean().values
    
    # ATR for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = max(PIVOT_LENGTH + 1, VOL_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if np.isnan(pivot_6h[i]) or np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= entry_price - ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
        elif position == -1:  # short position
            if close[i] >= entry_price + ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
                
        # Time-based exit
        if position != 0 and bars_since_entry >= MAX_HOLD_BARS:
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
            continue
            
        # Volume confirmation
        vol_confirmed = volume[i] > vol_ma[i] * VOL_BREAKOUT_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Breakout conditions
        breakout_long = close[i] > r3_6h[i] and vol_confirmed
        breakout_short = close[i] < s3_6h[i] and vol_confirmed
        
        # Enter new positions only if flat
        if position == 0:
            if breakout_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            elif breakout_short:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position * SIGNAL_SIZE
    
    return signals