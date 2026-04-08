#!/usr/bin/env python3
# 1d_weekly_donchian_breakout_volume_regime_v1
# Hypothesis: Daily timeframe strategy using weekly Donchian channel breakouts with volume confirmation and chop regime filter.
# Long: price breaks above weekly Donchian(20) high with volume > 1.5x 50-day average and chop < 61.8
# Short: price breaks below weekly Donchian(20) low with volume > 1.5x 50-day average and chop < 61.8
# Exit: price reverts to weekly Donchian midpoint or ATR-based stoploss (2.0x ATR)
# Uses 1d primary timeframe with 1w HTF for Donchian calculation.
# Target: 30-100 total trades over 4 years (7-25/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_weekly_donchian_breakout_volume_regime_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
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
    
    # Calculate volume ratio (current vs 50-period average)
    vol_sma = np.full(n, np.nan)
    for i in range(50, n):
        vol_sma[i] = np.mean(volume[i-50:i])
    vol_ratio = np.where(vol_sma > 0, volume / vol_sma, 0)
    
    # Calculate Choppiness Index (14-period)
    chop = np.full(n, np.nan)
    for i in range(14, n):
        atr_sum = np.sum(tr[i-14:i+1])
        highest_high = np.max(high[i-14:i+1])
        lowest_low = np.min(low[i-14:i+1])
        if highest_high > lowest_low and atr_sum > 0:
            chop[i] = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(14)
        else:
            chop[i] = 50.0  # neutral when undefined
    
    # Get 1w data for Donchian channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Donchian Channel (20-period) for each 1w bar
    donchian_high = np.full(len(df_1w), np.nan)
    donchian_low = np.full(len(df_1w), np.nan)
    donchian_mid = np.full(len(df_1w), np.nan)
    
    for i in range(len(df_1w)):
        if i < 19:
            continue
        donchian_high[i] = np.max(high_1w[i-19:i+1])
        donchian_low[i] = np.min(low_1w[i-19:i+1])
        donchian_mid[i] = (donchian_high[i] + donchian_low[i]) / 2.0
    
    # Align 1w Donchian levels to 1d timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1w, donchian_mid)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    atr_stop = 0.0
    
    for i in range(100, n):
        vol_r = vol_ratio[i]
        ch = chop[i]
        price = close[i]
        
        if np.isnan(vol_r) or np.isnan(ch) or np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or np.isnan(donchian_mid_aligned[i]) or np.isnan(atr[i]):
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
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reverts to midpoint OR stoploss hit (2.0x ATR above entry)
            if price >= donchian_mid_aligned[i] or price >= entry_price + 2.0 * atr_stop:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: price breaks above Donchian high with volume spike and chop filter
            if price > donchian_high_aligned[i] and vol_r > 1.5 and ch < 61.8:
                position = 1
                entry_price = price
                atr_stop = atr[i]
                signals[i] = 0.25
            # Short entry: price breaks below Donchian low with volume spike and chop filter
            elif price < donchian_low_aligned[i] and vol_r > 1.5 and ch < 61.8:
                position = -1
                entry_price = price
                atr_stop = atr[i]
                signals[i] = -0.25
    
    return signals