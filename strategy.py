#!/usr/bin/env python3
"""
4h_1d_camarilla_breakout_volatility
Breakout from Camarilla pivot levels (H4/L4) on 1d timeframe.
Entry on 4h when price closes beyond H4 (long) or L4 (short) with volume confirmation.
Exit when price returns to Pivot point or opposite Camarilla level.
Volatility filter: only trade when ATR(14) > 20-period SMA of ATR (avoid low volatility chop).
Designed for low trade frequency to minimize fee drag.
Works in both trending and mean-reverting markets by using volatility-adjusted breakouts.
"""

name = "4h_1d_camarilla_breakout_volatility"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    atr = np.zeros_like(tr)
    for i in range(len(tr)):
        if i < period:
            atr[i] = np.nan
        elif i == period:
            atr[i] = np.mean(tr[:period+1])
        else:
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Camarilla levels for each day
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # H4 = C + (Range * 1.1/2)
    # L4 = C - (Range * 1.1/2)
    # H3 = C + (Range * 1.1/4)
    # L3 = C - (Range * 1.1/4)
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    h4_1d = close_1d + (range_1d * 1.1 / 2)
    l4_1d = close_1d - (range_1d * 1.1 / 2)
    h3_1d = close_1d + (range_1d * 1.1 / 4)
    l3_1d = close_1d - (range_1d * 1.1 / 4)
    
    # Align Camarilla levels to 4h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4_1d)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4_1d)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    
    # Volatility filter: ATR > SMA of ATR (avoid low volatility chop)
    atr = calculate_atr(high, low, close, 14)
    atr_sma = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
    vol_filter = ~np.isnan(atr) & ~np.isnan(atr_sma) & (atr > atr_sma)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(pivot_aligned[i]) or np.isnan(h4_aligned[i]) or 
            np.isnan(l4_aligned[i]) or np.isnan(h3_aligned[i]) or 
            np.isnan(l3_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: price closes above H4 with volume and volatility
        if (close[i] > h4_aligned[i] and vol_confirm[i] and vol_filter[i] and 
            position != 1):
            position = 1
            signals[i] = 0.25
        # Short entry: price closes below L4 with volume and volatility
        elif (close[i] < l4_aligned[i] and vol_confirm[i] and vol_filter[i] and 
              position != -1):
            position = -1
            signals[i] = -0.25
        # Long exit: price returns to H3 or below Pivot
        elif position == 1 and (close[i] < h3_aligned[i] or close[i] < pivot_aligned[i]):
            position = 0
            signals[i] = 0.0
        # Short exit: price returns to L3 or above Pivot
        elif position == -1 and (close[i] > l3_aligned[i] or close[i] > pivot_aligned[i]):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals