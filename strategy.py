#!/usr/bin/env python3
# 12h_daily_camarilla_pivot_volume_regime_v2
# Hypothesis: 12h strategies using daily Camarilla pivot levels (H3/L3) with volume confirmation and 1d chop regime filter work in both bull and bear markets.
# Long: price breaks above H3 with volume > 1.8x 20-period average AND 1d chop > 61.8 (range regime)
# Short: price breaks below L3 with volume > 1.8x 20-period average AND 1d chop > 61.8 (range regime)
# Exit: price reverts to daily pivot (P) level
# Uses 12h primary timeframe with 1d HTF for Camarilla pivot and chop calculation.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_daily_camarilla_pivot_volume_regime_v2"
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
    
    # Calculate ATR(14) for stoploss (not used in signals but kept for potential exit logic)
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
    
    # Get 1d data for Camarilla pivot levels and chop regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for each 1d bar
    camarilla_p = np.full(len(df_1d), np.nan)
    camarilla_h3 = np.full(len(df_1d), np.nan)
    camarilla_l3 = np.full(len(df_1d), np.nan)
    
    for i in range(len(df_1d)):
        if np.isnan(high_1d[i]) or np.isnan(low_1d[i]) or np.isnan(close_1d[i]):
            continue
        diff = high_1d[i] - low_1d[i]
        camarilla_p[i] = (high_1d[i] + low_1d[i] + close_1d[i]) / 3.0
        camarilla_h3[i] = camarilla_p[i] + diff * 1.1 / 4.0
        camarilla_l3[i] = camarilla_p[i] - diff * 1.1 / 4.0
    
    # Calculate 1d Choppiness Index (CHOP) for regime filter
    # CHOP = 100 * log10(sum(ATR(14)) / (log(n) * (highest_high - lowest_low))) over period
    chop_period = 14
    tr_1d = np.zeros(len(df_1d))
    for i in range(1, len(df_1d)):
        tr_1d[i] = max(high_1d[i] - low_1d[i], abs(high_1d[i] - close_1d[i-1]), abs(low_1d[i] - close_1d[i-1]))
    
    atr_1d = np.full(len(df_1d), np.nan)
    for i in range(chop_period, len(df_1d)):
        atr_1d[i] = np.mean(tr_1d[i-chop_period:i])
    
    # Calculate rolling max/min for CHOP
    highest_high_1d = np.full(len(df_1d), np.nan)
    lowest_low_1d = np.full(len(df_1d), np.nan)
    for i in range(chop_period, len(df_1d)):
        highest_high_1d[i] = np.max(high_1d[i-chop_period:i+1])
        lowest_low_1d[i] = np.min(low_1d[i-chop_period:i+1])
    
    # Calculate CHOP: 100 * log10(sum(ATR) / (log(n) * (HH - LL)))
    chop_1d = np.full(len(df_1d), np.nan)
    for i in range(chop_period, len(df_1d)):
        sum_atr = np.sum(atr_1d[i-chop_period:i])
        hh_ll = highest_high_1d[i] - lowest_low_1d[i]
        if hh_ll > 0 and not np.isnan(sum_atr):
            chop_1d[i] = 100 * np.log10(sum_atr / (np.log(chop_period) * hh_ll))
        else:
            chop_1d[i] = 50  # neutral value
    
    # Align 1d data to 12h timeframe
    camarilla_p_aligned = align_htf_to_ltf(prices, df_1d, camarilla_p)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        vol_r = vol_ratio[i]
        price = close[i]
        
        # Skip if any required data is NaN
        if (np.isnan(vol_r) or np.isnan(camarilla_p_aligned[i]) or 
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(chop_1d_aligned[i])):
            if position != 0:
                # Hold current position
                signals[i] = 0.25 if position == 1 else -0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price reverts to pivot level
            if price <= camarilla_p_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reverts to pivot level
            if price >= camarilla_p_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Range regime: CHOP > 61.8 indicates ranging market (good for mean reversion at extremes)
            if chop_1d_aligned[i] > 61.8:
                # Long entry: price breaks above H3 with volume confirmation
                if price >= camarilla_h3_aligned[i] and vol_r > 1.8:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price breaks below L3 with volume confirmation
                elif price <= camarilla_l3_aligned[i] and vol_r > 1.8:
                    position = -1
                    signals[i] = -0.25
    
    return signals