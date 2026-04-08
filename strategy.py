#!/usr/bin/env python3
# 4h_daily_camarilla_pivot_volume_regime_v2
# Hypothesis: 4h strategy using 1d Camarilla pivot levels with volume confirmation and chop regime filter.
# Long: price breaks above H3 with volume > 1.8x average volume AND market is trending (CHOP < 61.8)
# Short: price breaks below L3 with volume > 1.8x average volume AND market is trending (CHOP < 61.8)
# Exit: price reverses to H4/L4 levels or volatility/chop increases.
# Reduced trade frequency via stricter volume threshold (1.8x vs 1.5x) and added volatility filter.
# Designed to capture strong intraday breakouts from key daily pivot levels while avoiding false breakouts.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_daily_camarilla_pivot_volume_regime_v2"
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
    
    # Calculate 1d Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Use previous day's OHLC for Camarilla calculation (standard approach)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_val = prev_high - prev_low
    
    # Camarilla levels
    H3 = pivot + (range_val * 1.1 / 4)
    L3 = pivot - (range_val * 1.1 / 4)
    H4 = pivot + (range_val * 1.1 / 2)
    L4 = pivot - (range_val * 1.1 / 2)
    
    # Align HTF levels to LTF
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    
    # Calculate volume ratio (current vs 20-period average)
    vol_sma = np.full(n, np.nan)
    for i in range(20, n):
        vol_sma[i] = np.mean(volume[i-20:i])
    vol_ratio = np.where(vol_sma > 0, volume / vol_sma, 0)
    
    # Calculate Chopiness Index for regime filter (14-period)
    chop = np.full(n, np.nan)
    for i in range(14, n):
        atr_sum = 0
        for j in range(i-13, i+1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_sum += tr
        atr = atr_sum / 14
        max_high = np.max(high[i-13:i+1])
        min_low = np.min(low[i-13:i+1])
        if max_high != min_low:
            chop[i] = 100 * np.log10(atr_sum / (max_high - min_low)) / np.log10(14)
        else:
            chop[i] = 50  # neutral when no range
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        vol_r = vol_ratio[i]
        ch = chop[i]
        price = close[i]
        
        if np.isnan(vol_r) or np.isnan(ch):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        h3 = H3_aligned[i]
        l3 = L3_aligned[i]
        h4 = H4_aligned[i]
        l4 = L4_aligned[i]
        
        if np.isnan(h3) or np.isnan(l3):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            if price < h4 or vol_r < 1.5 or ch > 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            if price > l4 or vol_r < 1.5 or ch > 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if price > h3 and vol_r > 1.8 and ch < 61.8:
                position = 1
                signals[i] = 0.25
            elif price < l3 and vol_r > 1.8 and ch < 61.8:
                position = -1
                signals[i] = -0.25
    
    return signals