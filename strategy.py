#!/usr/bin/env python3
"""
4h Camarilla R3/S3 Breakout with Volume Spike and Choppiness Filter.
Long when price breaks above R3 with volume spike and chop > 61.8 (ranging).
Short when price breaks below S3 with volume spike and chop > 61.8.
Exit when price crosses back below R3 (long) or above S3 (short).
Designed to generate 20-50 trades/year per symbol with mean-reversion edge in ranging markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given high, low, close"""
    range_val = high - low
    if range_val == 0:
        return close, close, close, close, close, close, close, close
    c = close
    h = high
    l = low
    # Camarilla levels
    r4 = c + range_val * 1.1 / 2
    r3 = c + range_val * 1.1 / 4
    r2 = c + range_val * 1.1 / 6
    r1 = c + range_val * 1.1 / 12
    s1 = c - range_val * 1.1 / 12
    s2 = c - range_val * 1.1 / 6
    s3 = c - range_val * 1.1 / 4
    s4 = c - range_val * 1.1 / 2
    return r3, r2, r1, c, s1, s2, s3, s4

def calculate_choppiness(high, low, close, period=14):
    """Calculate Choppiness Index"""
    n = len(high)
    cp = np.full(n, np.nan)
    if n < period:
        return cp
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # Sum of true range over period
    atr_sum = np.zeros(n)
    for i in range(period-1, n):
        atr_sum[i] = np.sum(tr[i-period+1:i+1])
    # Sum of absolute price change over period
    price_change = np.abs(np.diff(close))
    price_change_sum = np.zeros(n)
    for i in range(period-1, n):
        price_change_sum[i] = np.sum(price_change[i-period+1:i+1])
    # Choppiness formula
    for i in range(period-1, n):
        if atr_sum[i] > 0:
            cp[i] = 100 * np.log10(price_change_sum[i] / atr_sum[i]) / np.log10(period)
    return cp

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels and Choppiness
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each 1d bar
    n_1d = len(df_1d)
    r3_1d = np.full(n_1d, np.nan)
    s3_1d = np.full(n_1d, np.nan)
    
    for i in range(n_1d):
        r3, _, _, _, _, _, s3, _ = calculate_camarilla(
            df_1d['high'].iloc[i],
            df_1d['low'].iloc[i],
            df_1d['close'].iloc[i]
        )
        r3_1d[i] = r3
        s3_1d[i] = s3
    
    # Calculate Choppiness for each 1d bar
    chop_1d = calculate_choppiness(
        df_1d['high'].values,
        df_1d['low'].values,
        df_1d['close'].values,
        period=14
    )
    
    # Align to 4h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Volume filter: volume > 2.0x average (to avoid false signals)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Camarilla + chop + volume MA
    start_idx = max(19, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or 
            np.isnan(chop_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price and volume
        price_now = close[i]
        vol_now = volume[i]
        
        # Current levels
        r3 = r3_1d_aligned[i]
        s3 = s3_1d_aligned[i]
        chop = chop_1d_aligned[i]
        
        # Volume filter: volume > 2.0x average
        vol_filter = vol_now > 2.0 * vol_ma_20[i]
        
        # Choppiness filter: chop > 61.8 (ranging market)
        chop_filter = chop > 61.8
        
        if position == 0:
            # Long: price breaks above R3 + volume spike + chop filter
            if price_now > r3 and vol_filter and chop_filter:
                signals[i] = size
                position = 1
            # Short: price breaks below S3 + volume spike + chop filter
            elif price_now < s3 and vol_filter and chop_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses back below R3
            if price_now < r3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses back above S3
            if price_now > s3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_Volume_Chop"
timeframe = "4h"
leverage = 1.0