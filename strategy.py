#!/usr/bin/env python3
"""
6h Institutional Flow Detector with Volume Profile.
Long when price closes above Value Area High (VAH) with volume confirmation.
Short when price closes below Value Area Low (VAL) with volume confirmation.
Exit when price returns to Point of Control (POC).
Designed to capture institutional accumulation/distribution with low turnover.
"""

import numpy as np
import pandas as pd
from mpt_data import get_htf_data, align_htf_to_ltf

def calculate_vwap(high, low, close, volume):
    """Volume Weighted Average Price"""
    typical_price = (high + low + close) / 3.0
    vwap = np.cumsum(typical_price * volume) / np.cumsum(volume)
    return vwap

def calculate_value_area(high, low, close, volume, lookback=20):
    """Calculate Value Area High, Low, and Point of Control"""
    n = len(high)
    vah = np.full(n, np.nan)
    val = np.full(n, np.nan)
    poc = np.full(n, np.nan)
    
    if n < lookback:
        return vah, val, poc
    
    for i in range(lookback-1, n):
        start_idx = i - lookback + 1
        # Price bins: 100 bins between min and max
        price_min = np.min(low[start_idx:i+1])
        price_max = np.max(high[start_idx:i+1])
        if price_max <= price_min:
            continue
        bins = 100
        bin_width = (price_max - price_min) / bins
        volume_profile = np.zeros(bins)
        
        for j in range(start_idx, i+1):
            # Distribute volume across price range of the bar
            bar_low = low[j]
            bar_high = high[j]
            bar_volume = volume[j]
            if bar_high <= bar_low:
                continue
            # Simple allocation: volume at midpoint
            price_mid = (bar_low + bar_high) / 2.0
            bin_idx = int((price_mid - price_min) / bin_width)
            if 0 <= bin_idx < bins:
                volume_profile[bin_idx] += bar_volume
        
        # Point of Control: price with maximum volume
        if np.sum(volume_profile) > 0:
            poc_bin = np.argmax(volume_profile)
            poc[i] = price_min + poc_bin * bin_width + bin_width / 2.0
            
            # Value Area: 70% of volume around POC
            total_volume = np.sum(volume_profile)
            target_volume = 0.7 * total_volume
            
            # Expand outward from POC bin
            volume_accum = volume_profile[poc_bin]
            low_bin = poc_bin
            high_bin = poc_bin
            
            while volume_accum < target_volume and (low_bin > 0 or high_bin < bins - 1):
                # Expand to side with more volume
                vol_low = volume_profile[low_bin - 1] if low_bin > 0 else 0
                vol_high = volume_profile[high_bin + 1] if high_bin < bins - 1 else 0
                
                if vol_low >= vol_high and low_bin > 0:
                    low_bin -= 1
                    volume_accum += vol_low
                elif vol_high > vol_low and high_bin < bins - 1:
                    high_bin += 1
                    volume_accum += vol_high
                else:
                    break
            
            vah[i] = price_min + high_bin * bin_width + bin_width / 2.0
            val[i] = price_min + low_bin * bin_width + bin_width / 2.0
    
    return vah, val, poc

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Value Area calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Value Area on 1d data
    vah_1d, val_1d, poc_1d = calculate_value_area(
        df_1d['high'].values,
        df_1d['low'].values,
        df_1d['close'].values,
        df_1d['volume'].values,
        lookback=20
    )
    
    # Align Value Areas to 6h timeframe
    vah_aligned = align_htf_to_ltf(prices, df_1d, vah_1d)
    val_aligned = align_htf_to_ltf(prices, df_1d, val_1d)
    poc_aligned = align_htf_to_ltf(prices, df_1d, poc_1d)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need 20 periods for VA calculation + 20 for volume MA
    start_idx = max(39, 19)  # 20+20-1 for VA, 19 for volume MA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(vah_aligned[i]) or np.isnan(val_aligned[i]) or 
            np.isnan(poc_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price and volume
        price_now = close[i]
        vol_now = volume[i]
        
        # Current Value Area levels
        vah_val = vah_aligned[i]
        val_val = val_aligned[i]
        poc_val = poc_aligned[i]
        
        # Volume filter: volume > 1.5x average
        vol_filter = vol_now > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: price closes above VAH with volume confirmation
            if price_now > vah_val and vol_filter:
                signals[i] = size
                position = 1
            # Short: price closes below VAL with volume confirmation
            elif price_now < val_val and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to POC or below
            if price_now <= poc_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to POC or above
            if price_now >= poc_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Institutional_Flow_Detector"
timeframe = "6h"
leverage = 1.0