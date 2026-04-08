#!/usr/bin/env python3
# 1d_weekly_donchian_breakout_volume_regime_v1
# Hypothesis: 1d strategies based on weekly Donchian channel breakouts with volume confirmation and chop regime filter work in both bull and bear markets.
# Long: price breaks above weekly Donchian(20) high with volume > 1.5x 20-day average and chop < 61.8 (trending)
# Short: price breaks below weekly Donchian(20) low with volume > 1.5x 20-day average and chop < 61.8 (trending)
# Exit: price reverts to weekly Donchian midpoint or ATR-based stoploss (2.0x ATR)
# Uses 1d primary timeframe with 1w HTF for Donchian calculation and chop filter.
# Target: 30-100 total trades over 4 years (7-25/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_weekly_donchian_breakout_volume_regime_v1"
timeframe = "1d"
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
    
    # Get 1w data for Donchian channels and chop regime
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Donchian(20) channels
    donchian_high = np.full(len(df_1w), np.nan)
    donchian_low = np.full(len(df_1w), np.nan)
    donchian_mid = np.full(len(df_1w), np.nan)
    
    for i in range(20, len(df_1w)):
        donchian_high[i] = np.max(high_1w[i-20:i])
        donchian_low[i] = np.min(low_1w[i-20:i])
        donchian_mid[i] = (donchian_high[i] + donchian_low[i]) / 2.0
    
    # Calculate weekly Chopiness Index(14) for regime filter
    chop = np.full(len(df_1w), np.nan)
    for i in range(14, len(df_1w)):
        # True range
        tr1w = np.zeros(i+1)
        for j in range(1, i+1):
            tr1w[j] = max(high_1w[j] - low_1w[j], abs(high_1w[j] - close_1w[j-1]), abs(low_1w[j] - close_1w[j-1]))
        atr1w = np.mean(tr1w[-14:])
        # Sum of true ranges over 14 periods
        sum_tr14 = np.sum(tr1w[-14:])
        # Max high - min low over 14 periods
        max_high = np.max(high_1w[i-13:i+1])
        min_low = np.min(low_1w[i-13:i+1])
        range_maxmin = max_high - min_low
        if range_maxmin > 0:
            chop[i] = 100 * np.log10(sum_tr14 / range_maxmin) / np.log10(14)
        else:
            chop[i] = 50.0  # neutral when no range
    
    # Align 1w indicators to 1d timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1w, donchian_mid)
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    atr_stop = 0.0
    
    for i in range(50, n):
        vol_r = vol_ratio[i]
        price = close[i]
        
        # Skip if any required data is NaN
        if np.isnan(vol_r) or np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or np.isnan(donchian_mid_aligned[i]) or np.isnan(chop_aligned[i]) or np.isnan(atr[i]):
            # Hold current position if any, otherwise flat
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
            continue
        
        # Regime filter: only trade when chop < 61.8 (trending market)
        if chop_aligned[i] >= 61.8:
            # In ranging market, exit or stay flat
            if position == 1:
                if price <= donchian_mid_aligned[i] or price <= entry_price - 2.0 * atr_stop:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            elif position == -1:
                if price >= donchian_mid_aligned[i] or price >= entry_price + 2.0 * atr_stop:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
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
            # Long entry: price breaks above weekly Donchian high with volume spike and trending regime
            if price > donchian_high_aligned[i] and vol_r > 1.5:
                position = 1
                entry_price = price
                atr_stop = atr[i]
                signals[i] = 0.25
            # Short entry: price breaks below weekly Donchian low with volume spike and trending regime
            elif price < donchian_low_aligned[i] and vol_r > 1.5:
                position = -1
                entry_price = price
                atr_stop = atr[i]
                signals[i] = -0.25
    
    return signals