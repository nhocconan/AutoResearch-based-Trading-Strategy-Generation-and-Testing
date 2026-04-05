#!/usr/bin/env python3
"""
exp_7387_6d_pivot3levels_volatility_breakout_v1
Hypothesis: 6h strategy using daily pivot points (R3/S3 fade, R4/S4 breakout) with volatility filter.
Works in bull/bear via mean reversion at extremes and breakout continuation.
Target: 50-150 trades over 4 years. Uses discrete sizing (0.25) to minimize fee churn.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7387_6d_pivot3levels_volatility_breakout_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
PIVOT_LOOKBACK = 1  # Use previous day's OHLC for pivot
VOLATILITY_LOOKBACK = 24  # 24 * 6h = 6 days of volatility
VOLATILITY_THRESHOLD = 1.5  # Breakout when volatility > 1.5x average
PIVOT_BUFFER = 0.001  # 0.1% buffer around pivot levels
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
MAX_HOLD_BARS = 12  # Max 3 days

def calculate_pivot_levels(high, low, close):
    """Calculate classic pivot points: P, R1/S1, R2/S2, R3/S3, R4/S4"""
    pivot = (high + low + close) / 3.0
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    r3 = high + 2 * (pivot - low)
    s3 = low - 2 * (high - pivot)
    r4 = r3 + (high - low)
    s4 = s3 - (high - low)
    return pivot, r1, r2, r3, r4, s1, s2, s3, s4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1d for pivot levels
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily pivot levels from previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Arrays to store pivot levels
    pivot_vals = np.full_like(close_1d, np.nan)
    r3_vals = np.full_like(close_1d, np.nan)
    s3_vals = np.full_like(close_1d, np.nan)
    r4_vals = np.full_like(close_1d, np.nan)
    s4_vals = np.full_like(close_1d, np.nan)
    
    # Calculate pivots for each day (using previous day's data)
    for i in range(1, len(close_1d)):
        pivot, r1, r2, r3, r4, s1, s2, s3, s4 = calculate_pivot_levels(
            high_1d[i-1], low_1d[i-1], close_1d[i-1]
        )
        pivot_vals[i] = pivot
        r3_vals[i] = r3
        s3_vals[i] = s3
        r4_vals[i] = r4
        s4_vals[i] = s4
    
    # Align pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_vals)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_vals)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_vals)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4_vals)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4_vals)
    
    # Calculate 6h indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volatility measure: ATR-based
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr = pd.Series(tr).ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    # Average true range over longer period for volatility filter
    atr_ma = pd.Series(atr).rolling(window=VOLATILITY_LOOKBACK, min_periods=VOLATILITY_LOOKBACK).mean().values
    volatility_expanded = atr > (atr_ma * VOLATILITY_THRESHOLD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = max(VOLATILITY_LOOKBACK, ATR_PERIOD) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if np.isnan(pivot_aligned[i]):
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
            
        # Volatility filter - only trade when volatility is elevated
        vol_filter = volatility_expanded[i] if not np.isnan(atr_ma[i]) else False
        
        if vol_filter:
            # Fade at R3/S3 levels (mean reversion)
            fade_long = (close[i] <= s3_aligned[i] * (1 + PIVOT_BUFFER)) and (close[i] >= s3_aligned[i] * (1 - PIVOT_BUFFER))
            fade_short = (close[i] >= r3_aligned[i] * (1 - PIVOT_BUFFER)) and (close[i] <= r3_aligned[i] * (1 + PIVOT_BUFFER))
            
            # Breakout continuation at R4/S4 levels
            breakout_long = close[i] > r4_aligned[i]
            breakout_short = close[i] < s4_aligned[i]
            
            # Enter new positions only if flat
            if position == 0:
                if fade_long:
                    signals[i] = SIGNAL_SIZE
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                elif fade_short:
                    signals[i] = -SIGNAL_SIZE
                    position = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                elif breakout_long:
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
        else:
            # Low volatility - no trading
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
    
    return signals