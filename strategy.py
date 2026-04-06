#!/usr/bin/env python3
"""
6h Camarilla Pivot Reversal with Volume Confirmation
Hypothesis: Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) 
provide high-probability reversal/continuation signals when combined with volume spikes.
In ranging markets, price reverts from R3/S3. In trending markets, breaks of R4/S4 
continue the trend. Volume confirmation filters false signals. Works in both bull/bear.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14259_6h_camarilla_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def calculate_atr(high, low, close, period):
    """Calculate ATR with proper min_periods"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given period"""
    # Pivot point
    pivot = (high + low + close) / 3.0
    range_hl = high - low
    
    # Camarilla levels
    r4 = close + range_hl * 1.1 / 2
    r3 = close + range_hl * 1.1 / 4
    s3 = close - range_hl * 1.1 / 4
    s4 = close - range_hl * 1.1 / 2
    
    return pivot, r3, r4, s3, s4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data for Camarilla pivots (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily Camarilla levels
    camarilla_data = [calculate_camarilla(high_1d[i], low_1d[i], close_1d[i]) 
                      for i in range(len(close_1d))]
    pivot_1d = np.array([x[0] for x in camarilla_data])
    r3_1d = np.array([x[1] for x in camarilla_data])
    r4_1d = np.array([x[2] for x in camarilla_data])
    s3_1d = np.array([x[3] for x in camarilla_data])
    s4_1d = np.array([x[4] for x in camarilla_data])
    
    # Align to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: volume > 2.0x 24-period average (4 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_filter = volume > (2.0 * vol_ma)
    
    # ATR for stop loss (14-period)
    atr = calculate_atr(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period (max of 24 for volume, 14 for ATR)
    start = max(24, 14) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(pivot_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or \
           np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(atr[i]) or \
           np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
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
        
        # Camarilla signals with volume confirmation
        # Mean reversion: fade at R3/S3
        # Breakout: break and close beyond R4/S4
        fade_long = (close[i] <= s3_aligned[i]) and vol_filter[i]
        fade_short = (close[i] >= r3_aligned[i]) and vol_filter[i]
        breakout_long = (close[i] > r4_aligned[i]) and vol_filter[i]
        breakout_short = (close[i] < s4_aligned[i]) and vol_filter[i]
        
        # Generate signals
        if position == 0:
            if fade_long:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.0 * atr[i])
            elif fade_short:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.0 * atr[i])
            elif breakout_long:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.5 * atr[i])  # wider stop for breakout
            elif breakout_short:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.5 * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: stop hit OR mean reversion signal (price back to pivot) OR breakout failure
            if (close[i] <= stop_price or 
                close[i] >= pivot_aligned[i] or 
                (high[i] > r4_aligned[i] and close[i] < r4_aligned[i])):  # failed breakout
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: stop hit OR mean reversion signal OR breakout failure
            if (close[i] >= stop_price or 
                close[i] <= pivot_aligned[i] or 
                (low[i] < s4_aligned[i] and close[i] > s4_aligned[i])):  # failed breakout
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals