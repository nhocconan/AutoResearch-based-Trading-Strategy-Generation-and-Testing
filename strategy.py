#!/usr/bin/env python3
# 1h_4h_donchian_breakout_volume_regime_v1
# Hypothesis: 1h strategies with 4h Donchian breakout + volume confirmation + chop regime filter work in both bull and bear markets.
# Long: price breaks above 4h Donchian(20) high with volume > 1.5x 20-period average and CHOP(14) < 61.8 (trending)
# Short: price breaks below 4h Donchian(20) low with volume > 1.5x 20-period average and CHOP(14) < 61.8 (trending)
# Exit: price reverts to 4h Donchian midpoint or ATR-based stoploss (2.0x ATR)
# Uses 1h primary timeframe with 4h HTF for Donchian channels and chop regime.
# Target: 60-150 total trades over 4 years (15-37/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_donchian_breakout_volume_regime_v1"
timeframe = "1h"
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
    
    # Get 4h data for Donchian channels and chop regime
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h Donchian channels (20-period)
    donchian_high = np.full(len(df_4h), np.nan)
    donchian_low = np.full(len(df_4h), np.nan)
    donchian_mid = np.full(len(df_4h), np.nan)
    
    for i in range(20, len(df_4h)):
        donchian_high[i] = np.max(high_4h[i-20:i])
        donchian_low[i] = np.min(low_4h[i-20:i])
        donchian_mid[i] = (donchian_high[i] + donchian_low[i]) / 2.0
    
    # Calculate 4h Chopiness Index (14-period) for regime filter
    chop = np.full(len(df_4h), np.nan)
    for i in range(14, len(df_4h)):
        # True range sum
        tr_sum = 0.0
        for j in range(i-14, i):
            tr_j = max(high_4h[j] - low_4h[j], abs(high_4h[j] - close_4h[j-1]), abs(low_4h[j] - close_4h[j-1]))
            tr_sum += tr_j
        # Donchian width (max high - min low over period)
        max_high = np.max(high_4h[i-14:i])
        min_low = np.min(low_4h[i-14:i])
        donchian_width = max_high - min_low
        if donchian_width > 0 and tr_sum > 0:
            chop[i] = 100 * np.log10(tr_sum / donchian_width) / np.log10(14)
        else:
            chop[i] = 50.0  # neutral
    
    # Align 4h indicators to 1h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_4h, donchian_mid)
    chop_aligned = align_htf_to_ltf(prices, df_4h, chop)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    atr_stop = 0.0
    
    for i in range(50, n):
        vol_r = vol_ratio[i]
        price = close[i]
        chop_val = chop_aligned[i]
        
        if np.isnan(vol_r) or np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or np.isnan(donchian_mid_aligned[i]) or np.isnan(chop_val) or np.isnan(atr[i]):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price reverts to midpoint OR stoploss hit (2.0x ATR below entry)
            if price <= donchian_mid_aligned[i] or price <= entry_price - 2.0 * atr_stop:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price reverts to midpoint OR stoploss hit (2.0x ATR above entry)
            if price >= donchian_mid_aligned[i] or price >= entry_price + 2.0 * atr_stop:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Long entry: price breaks above Donchian high with volume spike and trending regime
            if price >= donchian_high_aligned[i] and vol_r > 1.5 and chop_val < 61.8:
                position = 1
                entry_price = price
                atr_stop = atr[i]
                signals[i] = 0.20
            # Short entry: price breaks below Donchian low with volume spike and trending regime
            elif price <= donchian_low_aligned[i] and vol_r > 1.5 and chop_val < 61.8:
                position = -1
                entry_price = price
                atr_stop = atr[i]
                signals[i] = -0.20
    
    return signals