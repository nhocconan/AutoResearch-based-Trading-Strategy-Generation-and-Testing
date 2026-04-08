#!/usr/bin/env python3
# [24966] 4h_1d_camarilla_pivot_volume_breakout_v1
# Hypothesis: 4-hour Camarilla pivot breakout with 1-day volume confirmation.
# Long when price breaks above R4 (strong resistance) with volume > 1.5x average and price > 20-period SMA.
# Short when price breaks below S4 (strong support) with volume > 1.5x average and price < 20-period SMA.
# Exit when price reverts to the pivot point (central level).
# Uses Camarilla levels from 1-day data for key institutional levels, effective in both trending and ranging markets.
# Target: 20-35 trades/year to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_pivot_volume_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-day data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    # Based on previous day's high, low, close
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_pivot = np.full(len(close_1d), np.nan)
    camarilla_r4 = np.full(len(close_1d), np.nan)  # Strong resistance
    camarilla_s4 = np.full(len(close_1d), np.nan)  # Strong support
    
    for i in range(1, len(close_1d)):
        # Use previous day's data
        high_y = high_1d[i-1]
        low_y = low_1d[i-1]
        close_y = close_1d[i-1]
        
        pivot = (high_y + low_y + close_y) / 3.0
        range_val = high_y - low_y
        
        camarilla_pivot[i] = pivot
        camarilla_r4[i] = pivot + (range_val * 1.1)  # R4 level
        camarilla_s4[i] = pivot - (range_val * 1.1)  # S4 level
    
    # Calculate 20-period SMA for trend filter
    sma_20 = np.full(n, np.nan)
    for i in range(20, n):
        sma_20[i] = np.mean(close[i-20:i])
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Align Camarilla levels to 4-hour timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(sma_20[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        price = close[i]
        
        if position == 1:  # Long
            # Exit: price returns to pivot point (mean reversion)
            if price <= pivot_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price returns to pivot point (mean reversion)
            if price >= pivot_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above R4 with volume expansion and above SMA20
            if price > r4_aligned[i] and vol_ratio > 1.5 and price > sma_20[i]:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below S4 with volume expansion and below SMA20
            elif price < s4_aligned[i] and vol_ratio > 1.5 and price < sma_20[i]:
                position = -1
                signals[i] = -0.25
    
    return signals