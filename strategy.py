#!/usr/bin/env python3
# Hypothesis: 4h Donchian breakout with 1-day ATR volatility filter and volume confirmation.
# Donchian channels identify breakouts with clear support/resistance levels.
# ATR filter ensures we only trade during sufficient volatility (avoid low-volatility whipsaws).
# Volume confirmation ensures breakouts have institutional participation.
# Designed for 4h timeframe to target 75-200 total trades over 4 years (19-50/year).
# Works in both bull and bear markets by filtering for volatility regimes.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for ATR volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:  # Need enough for ATR calculation
        return np.zeros(n)
    
    # Calculate daily ATR (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # ATR calculation using Wilder's smoothing (same as RSI)
    atr = np.zeros_like(tr)
    atr[0] = tr[0]
    for i in range(1, len(tr)):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # ATR ratio: current ATR / 50-period average ATR (volatility regime filter)
    atr_ma = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr / atr_ma
    atr_ratio[np.isnan(atr_ma) | (atr_ma == 0)] = 1.0  # Handle division by zero
    
    # Align ATR ratio to 4h timeframe
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Donchian channel on 4h data (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume filter: volume > 1.3x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback, 50)  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(atr_ratio_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: ATR ratio > 0.8 (avoid low volatility chop)
        vol_filter = atr_ratio_aligned[i] > 0.8
        
        # Breakout conditions
        long_breakout = close[i] > highest_high[i-1]  # Break above prior 20-period high
        short_breakout = close[i] < lowest_low[i-1]   # Break below prior 20-period low
        
        # Entry conditions with volume confirmation
        long_entry = vol_filter and long_breakout and volume_filter[i]
        short_entry = vol_filter and short_breakout and volume_filter[i]
        
        # Exit conditions: opposite breakout or volatility collapse
        long_exit = (close[i] < lowest_low[i-1]) or (atr_ratio_aligned[i] < 0.6)
        short_exit = (close[i] > highest_high[i-1]) or (atr_ratio_aligned[i] < 0.6)
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_DonchianBreakout_1dATR_VolFilter_Volume"
timeframe = "4h"
leverage = 1.0