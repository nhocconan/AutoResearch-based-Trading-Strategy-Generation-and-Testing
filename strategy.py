#!/usr/bin/env python3
"""
Hypothesis: 4h TRIX + volume spike + choppiness regime filter.
- Primary timeframe: 4h for execution, HTF: 1d for TRIX calculation and choppiness regime.
- TRIX (15,9) signals momentum: long when TRIX crosses above zero with volume spike and choppy market (CHOP>61.8),
  short when TRIX crosses below zero with volume spike and choppy market.
- Choppiness regime filter avoids trending markets where momentum fails.
- Volume spike confirms institutional participation.
- Works in bull via buying momentum in chop, in bear via selling momentum in chop.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_trix(close, period=15, signal=9):
    """Calculate TRIX indicator"""
    # Triple exponential smoothing
    ema1 = pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean()
    ema2 = ema1.ewm(span=period, adjust=False, min_periods=period).mean()
    ema3 = ema2.ewm(span=period, adjust=False, min_periods=period).mean()
    
    # TRIX = percentage change of ema3
    trix = ema3.pct_change() * 100
    
    # Signal line
    trix_signal = trix.ewm(span=signal, adjust=False, min_periods=signal).mean()
    
    return trix.values, trix_signal.values

def calculate_choppiness(high, low, close, period=14):
    """Calculate Choppiness Index"""
    atr_sum = 0.0
    for i in range(period):
        atr_sum += max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    highest_high = max(high[i - period + 1:i + 1]) if i >= period - 1 else high[i]
    lowest_low = min(low[i - period + 1:i + 1]) if i >= period - 1 else low[i]
    
    if highest_high == lowest_low:
        return 50.0
    
    chop = 100 * np.log10(atr_sum / np.log10(highest_high - lowest_low)) / np.log10(period)
    return chop

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for TRIX and choppiness
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d TRIX
    trix, trix_signal = calculate_trix(df_1d['close'].values, period=15, signal=9)
    
    # Calculate 1d Choppiness Index
    chop = np.full(len(df_1d), np.nan)
    for i in range(len(df_1d)):
        if i >= 13:  # Need at least 14 periods
            chop[i] = calculate_choppiness(
                df_1d['high'].iloc[i-13:i+1].values,
                df_1d['low'].iloc[i-13:i+1].values,
                df_1d['close'].iloc[i-13:i+1].values,
                period=14
            )
    
    # Align 1d indicators to 4h
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix)
    trix_signal_aligned = align_htf_to_ltf(prices, df_1d, trix_signal)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA (on 4h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(30, 20)  # Need enough 1d bars for TRIX and chop
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(trix_aligned[i]) or np.isnan(trix_signal_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Check for TRIX signals with volume spike and choppy regime (CHOP > 61.8 = ranging/choppy)
            if volume_spike[i] and chop_aligned[i] > 61.8:
                # Bullish signal: TRIX crosses above signal line
                if trix_aligned[i] > trix_signal_aligned[i] and trix_aligned[i-1] <= trix_signal_aligned[i-1]:
                    signals[i] = 0.25
                    position = 1
                # Bearish signal: TRIX crosses below signal line
                elif trix_aligned[i] < trix_signal_aligned[i] and trix_aligned[i-1] >= trix_signal_aligned[i-1]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: TRIX crosses below signal line or chop becomes too low (trending)
            if trix_aligned[i] < trix_signal_aligned[i] or chop_aligned[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: TRIX crosses above signal line or chop becomes too low (trending)
            if trix_aligned[i] > trix_signal_aligned[i] or chop_aligned[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_TRIX_VolumeSpike_ChopRegime_v1"
timeframe = "4h"
leverage = 1.0