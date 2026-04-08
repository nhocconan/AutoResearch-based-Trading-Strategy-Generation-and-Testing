#!/usr/bin/env python3
# 12h_weekly_donchian_breakout_volume_v1
# Hypothesis: 12h strategies based on weekly Donchian channel breakouts with volume confirmation work in both bull and bear markets.
# Long: price breaks above weekly Donchian(20) upper band with volume > 1.5x 20-period average
# Short: price breaks below weekly Donchian(20) lower band with volume > 1.5x 20-period average
# Exit: price reverts to weekly Donchian midpoint or ATR-based stoploss (2.0x ATR)
# Uses 12h primary timeframe with 1w HTF for Donchian calculation.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_weekly_donchian_breakout_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ATR(14) for stoploss
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr = np.full(n, np.nan)
    for i in range(14, n):
        atr[i] = np.mean(tr[i-14:i])
    
    # Calculate volume ratio (current vs 20-period average)
    vol_sma = np.full(n, np.nan)
    for i in range(20, n):
        vol_sma[i] = np.mean(volume[i-20:i])
    vol_ratio = np.where(vol_sma > 0, volume / vol_sma, 0)
    
    # Get 1w data for Donchian channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Donchian(20) channels
    upper_20 = np.full(len(df_1w), np.nan)
    lower_20 = np.full(len(df_1w), np.nan)
    midpoint_20 = np.full(len(df_1w), np.nan)
    
    for i in range(len(df_1w)):
        if i < 20:
            continue
        upper_20[i] = np.max(high_1w[i-20:i])
        lower_20[i] = np.min(low_1w[i-20:i])
        midpoint_20[i] = (upper_20[i] + lower_20[i]) / 2.0
    
    # Align 1w Donchian levels to 12h timeframe
    upper_20_aligned = align_htf_to_ltf(prices, df_1w, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1w, lower_20)
    midpoint_20_aligned = align_htf_to_ltf(prices, df_1w, midpoint_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    atr_stop = 0.0
    
    for i in range(50, n):
        vol_r = vol_ratio[i]
        price = close[i]
        
        if np.isnan(vol_r) or np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or np.isnan(midpoint_20_aligned[i]) or np.isnan(atr[i]):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price reverts to midpoint OR stoploss hit (2.0x ATR below entry)
            if price <= midpoint_20_aligned[i] or price <= entry_price - 2.0 * atr_stop:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reverts to midpoint OR stoploss hit (2.0x ATR above entry)
            if price >= midpoint_20_aligned[i] or price >= entry_price + 2.0 * atr_stop:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: price breaks above upper band with volume confirmation
            if price > upper_20_aligned[i] and vol_r > 1.5:
                position = 1
                entry_price = price
                atr_stop = atr[i]
                signals[i] = 0.25
            # Short entry: price breaks below lower band with volume confirmation
            elif price < lower_20_aligned[i] and vol_r > 1.5:
                position = -1
                entry_price = price
                atr_stop = atr[i]
                signals[i] = -0.25
    
    return signals