#!/usr/bin/env python3
# 4h_daily_camarilla_pivot_volume_spike_v5
# Hypothesis: Refine Camarilla pivot strategy with stricter volume confirmation (3.0x) and ATR-based exits (2.0x) to reduce overtrading and improve Sharpe.
# Long: price breaks above Camarilla H3 with volume > 3.0x 20-period average
# Short: price breaks below Camarilla L3 with volume > 3.0x 20-period average
# Exit: price reverts to Camarilla Pivot (midpoint) or ATR stoploss (2.0x ATR)
# Uses 4h primary timeframe with 1d HTF for Camarilla pivot calculation.
# Target: 75-150 total trades over 4 years to minimize fee drag and avoid overtrading.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_daily_camarilla_pivot_volume_spike_v5"
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
    
    # Calculate ATR(14) for stoploss with min_periods
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr = np.full(n, np.nan)
    for i in range(14, n):
        atr[i] = np.mean(tr[i-14:i])
    
    # Calculate volume ratio (current vs 20-period average) with min_periods
    vol_sma = np.full(n, np.nan)
    for i in range(20, n):
        vol_sma[i] = np.mean(volume[i-20:i])
    vol_ratio = np.where(vol_sma > 0, volume / vol_sma, 0)
    
    # Get 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for each 1d bar
    camarilla_p = np.full(len(df_1d), np.nan)
    camarilla_h3 = np.full(len(df_1d), np.nan)
    camarilla_l3 = np.full(len(df_1d), np.nan)
    camarilla_h4 = np.full(len(df_1d), np.nan)
    camarilla_l4 = np.full(len(df_1d), np.nan)
    
    for i in range(len(df_1d)):
        if i == 0 or np.isnan(high_1d[i]) or np.isnan(low_1d[i]) or np.isnan(close_1d[i]):
            continue
        diff = high_1d[i] - low_1d[i]
        camarilla_p[i] = (high_1d[i] + low_1d[i] + close_1d[i]) / 3.0
        camarilla_h3[i] = camarilla_p[i] + diff * 1.1 / 4.0
        camarilla_l3[i] = camarilla_p[i] - diff * 1.1 / 4.0
        camarilla_h4[i] = camarilla_p[i] + diff * 1.1 / 2.0
        camarilla_l4[i] = camarilla_p[i] - diff * 1.1 / 2.0
    
    # Align 1d Camarilla levels to 4h timeframe
    camarilla_p_aligned = align_htf_to_ltf(prices, df_1d, camarilla_p)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    atr_stop = 0.0
    bars_since_entry = 0
    
    for i in range(50, n):
        vol_r = vol_ratio[i]
        price = close[i]
        
        # Skip if any required data is NaN
        if np.isnan(vol_r) or np.isnan(camarilla_p_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or np.isnan(atr[i]):
            # Hold current position if any, otherwise flat
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
            continue
        
        if position == 1:  # Long position
            bars_since_entry += 1
            # Exit: price reverts to pivot OR stoploss hit (2.0x ATR below entry) OR min holding period (4 bars) passed
            if (price <= camarilla_p_aligned[i] or 
                price <= entry_price - 2.0 * atr_stop or 
                bars_since_entry >= 4):
                position = 0
                bars_since_entry = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            bars_since_entry += 1
            # Exit: price reverts to pivot OR stoploss hit (2.0x ATR above entry) OR min holding period (4 bars) passed
            if (price >= camarilla_p_aligned[i] or 
                price >= entry_price + 2.0 * atr_stop or 
                bars_since_entry >= 4):
                position = 0
                bars_since_entry = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            bars_since_entry = 0
            # Long entry: price breaks above H3 with volume spike
            if price > camarilla_h3_aligned[i] and vol_r > 3.0:
                position = 1
                entry_price = price
                atr_stop = atr[i]
                signals[i] = 0.25
            # Short entry: price breaks below L3 with volume spike
            elif price < camarilla_l3_aligned[i] and vol_r > 3.0:
                position = -1
                entry_price = price
                atr_stop = atr[i]
                signals[i] = -0.25
    
    return signals