#!/usr/bin/env python3
"""
6h_1d_Pivot_R1S1_Breakout_VolumeATRFilter
Hypothesis: Breakouts above daily R1 or below daily S1 on 6h timeframe with volume confirmation (2.0x ATR-scaled volume) and ATR filter (current ATR < 1.5x 20-period ATR mean) capture high-probability momentum moves while avoiding choppy periods. Works in bull/bear markets by taking breakouts in direction of price action with volatility-adjusted filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range (ATR)"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = np.zeros_like(tr)
    if len(tr) >= period:
        atr[period-1] = np.mean(tr[:period])
    
    for i in range(period, len(tr)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    
    return atr

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given high, low, close"""
    range_val = high - low
    if range_val == 0:
        return close, close
    r1 = close + range_val * 1.1 / 12
    s1 = close - range_val * 1.1 / 12
    return r1, s1

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data once for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily OHLC arrays
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR for volatility filter
    atr = calculate_atr(prices['high'].values, prices['low'].values, prices['close'].values, 14)
    
    # Align daily OHLC to 6h timeframe
    high_1d_aligned = align_htf_to_ltf(prices, df_1d, high_1d)
    low_1d_aligned = align_htf_to_ltf(prices, df_1d, low_1d)
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if ATR not ready
        if np.isnan(atr[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate daily Camarilla levels from previous day's OHLC
        # Use prior bar's aligned daily values (previous completed day)
        prev_high = high_1d_aligned[i-1]
        prev_low = low_1d_aligned[i-1]
        prev_close = close_1d_aligned[i-1]
        
        r1, s1 = calculate_camarilla(prev_high, prev_low, prev_close)
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation: volume > 2.0 * ATR (volatility-adjusted)
        vol_threshold = 2.0 * atr[i] * 100  # Scale ATR to approximate volume units
        volume_ok = volume > vol_threshold
        
        # ATR filter: current ATR < 1.5 * 20-period ATR mean (avoid high volatility/chop)
        if i >= 20:
            atr_ma = np.mean(atr[i-20:i])
            atr_filter = atr[i] < 1.5 * atr_ma
        else:
            atr_filter = True
        
        if position == 0:
            # Long: price breaks above R1 + volume confirmation + ATR filter
            if price > r1 and volume_ok and atr_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + volume confirmation + ATR filter
            elif price < s1 and volume_ok and atr_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below S1 or ATR expands significantly (chop incoming)
            if price < s1 or atr[i] > 2.0 * atr_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above R1 or ATR expands significantly (chop incoming)
            if price > r1 or atr[i] > 2.0 * atr_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1d_Pivot_R1S1_Breakout_VolumeATRFilter"
timeframe = "6h"
leverage = 1.0