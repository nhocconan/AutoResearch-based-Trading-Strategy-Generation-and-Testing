#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Institutional Flow Detector using 1d VWAP bands and volume imbalance
# Detects when smart money is accumulating/distributing by comparing price action to
# volume-weighted average price (VWAP) bands. Works in bull/bear by following
# institutional flow direction confirmed by volume spikes.
# Target: 50-150 total trades over 4 years (~12-37/year) to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for VWAP calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d VWAP
    vwap_1d = np.full(len(df_1d), np.nan)
    for i in range(len(df_1d)):
        typical_price = (high_1d[i] + low_1d[i] + close_1d[i]) / 3.0
        vwap_1d[i] = typical_price * volume_1d[i]
        if i > 0:
            vwap_1d[i] += vwap_1d[i-1]
        # Cumulative VWAP
        cum_vol = np.sum(volume_1d[:i+1])
        if cum_vol > 0:
            vwap_1d[i] = vwap_1d[i] / cum_vol
        else:
            vwap_1d[i] = typical_price
    
    # Calculate 1d VWAP bands (1.5 * ATR)
    atr_1d = np.full(len(df_1d), np.nan)
    for i in range(1, len(df_1d)):
        tr = max(
            high_1d[i] - low_1d[i],
            abs(high_1d[i] - close_1d[i-1]),
            abs(low_1d[i] - close_1d[i-1])
        )
        if i == 1:
            atr_1d[i] = tr
        else:
            atr_1d[i] = 0.9 * atr_1d[i-1] + 0.1 * tr
    
    vwap_upper_1d = vwap_1d + 1.5 * atr_1d
    vwap_lower_1d = vwap_1d - 1.5 * atr_1d
    
    # Align VWAP and bands to 6h timeframe
    vwap_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    vwap_upper_aligned = align_htf_to_ltf(prices, df_1d, vwap_upper_1d)
    vwap_lower_aligned = align_htf_to_ltf(prices, df_1d, vwap_lower_1d)
    
    # Volume imbalance: current 6h volume vs average
    vol_ma_12 = np.full(n, np.nan)  # 2 periods of 6h = 12h
    for i in range(11, n):
        vol_ma_12[i] = np.mean(volume[i-11:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need 1d VWAP (1 bar), ATR (2), volume MA (12)
    start_idx = max(1, 2, 11)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(vwap_aligned[i]) or np.isnan(vwap_upper_aligned[i]) or 
            np.isnan(vwap_lower_aligned[i]) or np.isnan(vol_ma_12[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_12[i]
        
        # Volume filter: above average volume
        vol_filter = vol_now > 1.5 * vol_avg
        
        # Price position relative to VWAP bands
        price_vs_vwap = (price - vwap_aligned[i]) / vwap_aligned[i]
        
        if position == 0:
            # Long: price below lower VWAP band with volume (accumulation)
            if price < vwap_lower_aligned[i] and vol_filter:
                signals[i] = size
                position = 1
            # Short: price above upper VWAP band with volume (distribution)
            elif price > vwap_upper_aligned[i] and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to VWAP or volume dries up
            if price >= vwap_aligned[i] or vol_now < vol_avg:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to VWAP or volume dries up
            if price <= vwap_aligned[i] or vol_now < vol_avg:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Institutional_Flow_Detector_1dVWAP_Volume"
timeframe = "6h"
leverage = 1.0