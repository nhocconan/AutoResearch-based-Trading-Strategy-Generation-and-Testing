#!/usr/bin/env python3
# 4h_12h_Camarilla_Volume_Chop_v1
# Hypothesis: Use 12h Camarilla pivot levels with 4h volume confirmation and 12h Choppiness index regime filter.
# Long when price touches L3 with bullish 12h regime and volume spike, short when price touches H3 with bearish 12h regime and volume spike.
# Works in bull/bear by fading extremes in ranging markets and avoiding trending regimes.
# Target: 80-150 total trades over 4 years (20-38/year).

name = "4h_12h_Camarilla_Volume_Chop_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given period"""
    pivot = (high + low + close) / 3
    range_val = high - low
    h4 = pivot + (range_val * 1.1 / 2)
    h3 = pivot + (range_val * 1.1 / 4)
    l3 = pivot - (range_val * 1.1 / 4)
    l4 = pivot - (range_val * 1.1 / 2)
    return h3, l3

def calculate_chop(high, low, close, period=14):
    """Calculate Choppiness Index"""
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # First TR uses previous close, handle index 0
    tr[0] = high[0] - low[0]
    
    atr = np.zeros_like(close)
    for i in range(len(close)):
        if i < period:
            atr[i] = np.mean(tr[max(0, i-period+1):i+1])
        else:
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    
    # Sum of TR over period
    tr_sum = np.zeros_like(close)
    for i in range(len(close)):
        if i < period:
            tr_sum[i] = np.sum(tr[max(0, i-period+1):i+1])
        else:
            tr_sum[i] = tr_sum[i-1] - tr[i-period] + tr[i]
    
    # Chop = 100 * log10(tr_sum / (atr * period)) / log10(period)
    chop = 100 * np.log10(tr_sum / (atr * period)) / np.log10(period)
    return chop

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Camarilla and Chop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h Camarilla levels
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    h3_12h, l3_12h = calculate_camarilla(high_12h, low_12h, close_12h)
    
    # Calculate 12h Choppiness Index
    chop_12h = calculate_chop(high_12h, low_12h, close_12h)
    
    # Align 12h indicators to 4h
    h3_12h_aligned = align_htf_to_ltf(prices, df_12h, h3_12h)
    l3_12h_aligned = align_htf_to_ltf(prices, df_12h, l3_12h)
    chop_12h_aligned = align_htf_to_ltf(prices, df_12h, chop_12h)
    
    # 4h volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    
    start_idx = max(20, 20) + 1  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(h3_12h_aligned[i]) or np.isnan(l3_12h_aligned[i]) or 
            np.isnan(chop_12h_aligned[i]) or np.isnan(vol_ma[i])):
            continue
        
        # Regime filter: Chop > 61.8 = ranging (good for mean reversion)
        # Chop < 38.2 = trending (avoid)
        if chop_12h_aligned[i] < 38.2:  # Trending regime - avoid
            continue
        
        # Long setup: price touches L3 with volume spike in ranging market
        if low[i] <= l3_12h_aligned[i] and vol_spike[i]:
            signals[i] = 0.25
        
        # Short setup: price touches H3 with volume spike in ranging market
        elif high[i] >= h3_12h_aligned[i] and vol_spike[i]:
            signals[i] = -0.25
    
    return signals