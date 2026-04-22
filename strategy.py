#!/usr/bin/env python3

"""
Hypothesis: 4-hour TRIX momentum with volume confirmation and Choppiness regime filter.
Goes long when TRIX crosses above zero line (momentum shift up) in trending markets (CHOPPINESS < 38.2),
short when TRIX crosses below zero in trending markets. Uses volume spike to confirm institutional participation.
Designed for low trade frequency (20-50 trades/year) to minimize fee drift and work in both bull and bear markets
by avoiding ranging conditions (CHOPPINESS > 61.8) where momentum fails.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_trix(close, period=12):
    """Calculate TRIX: triple smoothed EMA rate of change."""
    ema1 = pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean()
    ema2 = ema1.ewm(span=period, adjust=False, min_periods=period).mean()
    ema3 = ema2.ewm(span=period, adjust=False, min_periods=period).mean()
    trix = ema3.pct_change(periods=1) * 100
    return trix.values

def calculate_chop(high, low, close, period=14):
    """Calculate Choppiness Index: measures market consolidation vs trending."""
    atr = pd.Series(np.sqrt(((high - low)**2 + (high - close.shift(1))**2 + (low - close.shift(1))**2) / 3)).rolling(window=period, min_periods=period).mean()
    max_high = pd.Series(high).rolling(window=period, min_periods=period).max()
    min_low = pd.Series(low).rolling(window=period, min_periods=period).min()
    chop = 100 * np.log10(atr.sum() / (max_high - min_low)) / np.log10(period)
    return chop.values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data for Choppiness filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Daily Choppiness for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    chop_1d = calculate_chop(high_1d, low_1d, close_1d, period=14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # TRIX indicator on price
    trix = calculate_trix(close, period=12)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(trix[i]) or np.isnan(chop_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        # Trending market condition (Choppiness < 38.2)
        trending = chop_1d_aligned[i] < 38.2
        
        if position == 0 and vol_spike and trending:
            # Long: TRIX crosses above zero with volume
            if trix[i] > 0 and trix[i-1] <= 0:
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below zero with volume
            elif trix[i] < 0 and trix[i-1] >= 0:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: TRIX crosses zero in opposite direction or market becomes ranging
            exit_signal = False
            
            if position == 1:
                # Exit long: TRIX crosses below zero or market ranges
                if trix[i] < 0 and trix[i-1] >= 0 or chop_1d_aligned[i] > 61.8:
                    exit_signal = True
            else:  # position == -1
                # Exit short: TRIX crosses above zero or market ranges
                if trix[i] > 0 and trix[i-1] <= 0 or chop_1d_aligned[i] > 61.8:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_TRIX_ZeroCross_Volume_ChopFilter"
timeframe = "4h"
leverage = 1.0