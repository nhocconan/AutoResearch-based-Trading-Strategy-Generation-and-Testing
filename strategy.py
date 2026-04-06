#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot levels from 1d: fade at R3/S3, breakout continuation at R4/S4
# Camarilla levels provide intraday support/resistance with high probability of mean reversion at R3/S3
# and breakout continuation at R4/S4. Works in both bull and bear markets as it adapts to price action.
# Target: 80-180 total trades over 4 years (20-45/year).

name = "exp_13571_6h_camarilla1d_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
PIVOT_LOOKBACK = 1
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
CAMARILLA_MULTIPLIER = 1.1  # Standard Camarilla multiplier

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given high, low, close"""
    pivot = (high + low + close) / 3.0
    range_ = high - low
    r4 = pivot + (range_ * CAMARILLA_MULTIPLIER * 1.5)
    r3 = pivot + (range_ * CAMARILLA_MULTIPLIER * 1.25)
    s3 = pivot - (range_ * CAMARILLA_MULTIPLIER * 1.25)
    s4 = pivot - (range_ * CAMARILLA_MULTIPLIER * 1.5)
    return pivot, r3, r4, s3, s4

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
    
    # Calculate daily Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot_1d = np.zeros(len(close_1d))
    r3_1d = np.zeros(len(close_1d))
    r4_1d = np.zeros(len(close_1d))
    s3_1d = np.zeros(len(close_1d))
    s4_1d = np.zeros(len(close_1d))
    
    for i in range(len(close_1d)):
        if i < PIVOT_LOOKBACK:
            pivot_1d[i] = np.nan
            r3_1d[i] = np.nan
            r4_1d[i] = np.nan
            s3_1d[i] = np.nan
            s4_1d[i] = np.nan
        else:
            idx = i - PIVOT_LOOKBACK
            pivot, r3, r4, s3, s4 = calculate_camarilla(high_1d[idx], low_1d[idx], close_1d[idx])
            pivot_1d[i] = pivot
            r3_1d[i] = r3
            r4_1d[i] = r4
            s3_1d[i] = s3
            s4_1d[i] = s4
    
    # Align Camarilla levels to 6h timeframe
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Calculate 6h ATR for stop loss
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = PIVOT_LOOKBACK + 1
    
    for i in range(start, n):
        # Skip if Camarilla levels not available
        if np.isnan(pivot_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or np.isnan(r4_1d_aligned[i]) or \
           np.isnan(s3_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]):
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
        
        # Fade at R3/S3 (mean reversion)
        fade_short = close[i] >= r3_1d_aligned[i] and close[i] <= r4_1d_aligned[i]
        fade_long = close[i] <= s3_1d_aligned[i] and close[i] >= s4_1d_aligned[i]
        
        # Breakout continuation at R4/S4
        breakout_long = close[i] > r4_1d_aligned[i]
        breakout_short = close[i] < s4_1d_aligned[i]
        
        # Generate signals
        if position == 0:
            if fade_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif fade_short:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            elif breakout_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif breakout_short:
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