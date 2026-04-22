#!/usr/bin/env python3
"""
Hypothesis: 6-hour MACD with 1-day ADX filter and volume confirmation.
Long when MACD line crosses above signal line, ADX > 25 (trending), and volume > 50-period average volume.
Short when MACD line crosses below signal line, ADX > 25, and volume > 50-period average volume.
Exit when MACD reverses or volume drops below average.
MACD captures momentum, ADX ensures trending markets, volume confirms institutional participation.
Works in both bull and bear markets by following momentum in trending regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # MACD calculation (12,26,9)
    close_series = pd.Series(close)
    ema12 = close_series.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = close_series.ewm(span=26, adjust=False, min_periods=26).mean()
    macd_line = (ema12 - ema26).values
    signal_line = pd.Series(macd_line).ewm(span=9, adjust=False, min_periods=9).mean().values
    macd_histogram = macd_line - signal_line
    
    # Load 1-day data for ADX and volume filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # ADX calculation (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    tr14 = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / tr14
    di_minus = 100 * dm_minus_14 / tr14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx = np.concatenate([np.full(14, np.nan), adx[14:]])  # First 14 values are NaN
    
    # Volume average
    volume_1d = df_1d['volume'].values
    avg_vol_1d = pd.Series(volume_1d).rolling(window=50, min_periods=50).mean().values
    
    # Align HTF data
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    avg_vol_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(macd_line[i]) or np.isnan(signal_line[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(avg_vol_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: MACD bullish crossover, ADX > 25, volume above average
            if (macd_line[i] > signal_line[i] and macd_line[i-1] <= signal_line[i-1] and
                adx_aligned[i] > 25 and volume[i] > avg_vol_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: MACD bearish crossover, ADX > 25, volume above average
            elif (macd_line[i] < signal_line[i] and macd_line[i-1] >= signal_line[i-1] and
                  adx_aligned[i] > 25 and volume[i] > avg_vol_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: MACD bearish crossover
                if macd_line[i] < signal_line[i] and macd_line[i-1] >= signal_line[i-1]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: MACD bullish crossover
                if macd_line[i] > signal_line[i] and macd_line[i-1] <= signal_line[i-1]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_MACD_1dADX_Volume_Filter"
timeframe = "6h"
leverage = 1.0