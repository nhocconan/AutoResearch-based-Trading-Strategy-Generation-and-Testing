#!/usr/bin/env python3
"""
4h_1d_donchian_volume_crossover_v1
Hypothesis: 4h price crossing above/below 1d Donchian channels with volume confirmation.
Long when 4h close > 1d Donchian high + volume spike; short when 4h close < 1d Donchian low + volume spike.
Uses 1d ATR for volatility filter to avoid false breakouts in low volatility.
Designed for low trade frequency (20-40/year) to minimize fee drag.
Works in bull/bear via volatility filter and volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_donchian_volume_crossover_v1"
timeframe = "4h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing"""
    if len(high) < period:
        return np.full_like(high, np.nan, dtype=float)
    
    tr = np.zeros(len(high))
    tr[0] = high[0] - low[0]
    for i in range(1, len(high)):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = np.full_like(high, np.nan, dtype=float)
    atr[period-1] = np.mean(tr[:period])
    for i in range(period, len(high)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian channels and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    donchian_high = np.full(len(close_1d), np.nan)
    donchian_low = np.full(len(close_1d), np.nan)
    
    for i in range(20, len(close_1d)):
        donchian_high[i] = np.max(high_1d[i-20:i])
        donchian_low[i] = np.min(low_1d[i-20:i])
    
    # Calculate 1d ATR (14-period) for volatility filter
    atr_1d = calculate_atr(high_1d, low_1d, close_1d, 14)
    
    # Align indicators to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Volume confirmation: 20-period average on 4h
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(atr_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        price = close[i]
        upper = donchian_high_aligned[i]
        lower = donchian_low_aligned[i]
        atr_val = atr_1d_aligned[i]
        
        if position == 1:  # Long
            # Exit: price closes below Donchian low or volatility contracts
            if price < lower or vol_ratio < 0.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price closes above Donchian high or volatility contracts
            if price > upper or vol_ratio < 0.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: break above Donchian high with volume expansion and sufficient volatility
            if price > upper and vol_ratio > 1.5 and atr_val > 0.5 * np.nanmedian(atr_1d_aligned[max(0, i-20):i]):
                position = 1
                signals[i] = 0.25
            # Enter short: break below Donchian low with volume expansion and sufficient volatility
            elif price < lower and vol_ratio > 1.5 and atr_val > 0.5 * np.nanmedian(atr_1d_aligned[max(0, i-20):i]):
                position = -1
                signals[i] = -0.25
    
    return signals