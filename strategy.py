#!/usr/bin/env python3
# 4h_daily_camarilla_pivot_volume_regime_v3
# Hypothesis: 4h mean reversion at Camarilla pivot levels (S3/S4 for long, R3/R4 for short) from 1d HTF,
# with volume confirmation (>1.5x average) and 1d chop regime filter (chop < 61.8 = trending, chop > 61.8 = ranging).
# In trending markets (chop < 61.8): trade breakouts of R4/S4. In ranging markets (chop > 61.8): trade mean reversion at R3/S3.
# Uses 4h primary timeframe with 1d HTF for pivot levels and regime filter to reduce overtrading.
# Target: 75-200 trades over 4 years (19-50/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_daily_camarilla_pivot_volume_regime_v3"
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
    
    # Calculate volume ratio (current vs 20-period average)
    vol_sma = np.full(n, np.nan)
    for i in range(20, n):
        vol_sma[i] = np.mean(volume[i-20:i])
    vol_ratio = np.where(vol_sma > 0, volume / vol_sma, 0)
    
    # Get 1d data for Camarilla pivots and chop regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_r3 = np.full(len(df_1d), np.nan)
    camarilla_r4 = np.full(len(df_1d), np.nan)
    camarilla_s3 = np.full(len(df_1d), np.nan)
    camarilla_s4 = np.full(len(df_1d), np.nan)
    pivot = np.full(len(df_1d), np.nan)
    
    for i in range(len(df_1d)):
        if np.isnan(high_1d[i]) or np.isnan(low_1d[i]) or np.isnan(close_1d[i]):
            continue
        pivot[i] = (high_1d[i] + low_1d[i] + close_1d[i]) / 3
        range_1d = high_1d[i] - low_1d[i]
        camarilla_r4[i] = close_1d[i] + range_1d * 1.1 / 2
        camarilla_r3[i] = close_1d[i] + range_1d * 1.1 / 4
        camarilla_s3[i] = close_1d[i] - range_1d * 1.1 / 4
        camarilla_s4[i] = close_1d[i] - range_1d * 1.1 / 2
    
    # Calculate Chopiness Index on 1d data (14-period)
    chop_1d = np.full(len(df_1d), np.nan)
    for i in range(14, len(df_1d)):
        atr_sum = 0
        for j in range(i-13, i+1):
            tr = max(high_1d[j] - low_1d[j],
                     abs(high_1d[j] - close_1d[j-1]),
                     abs(low_1d[j] - close_1d[j-1]))
            atr_sum += tr
        atr = atr_sum / 14
        max_high = np.max(high_1d[i-13:i+1].values)
        min_low = np.min(low_1d[i-13:i+1].values)
        if max_high != min_low:
            chop_1d[i] = 100 * np.log10(atr_sum / (max_high - min_low)) / np.log10(14)
        else:
            chop_1d[i] = 50  # neutral when no range
    
    # Align 1d data to 4h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        vol_r = vol_ratio[i]
        ch = chop_aligned[i]
        price = close[i]
        
        if np.isnan(vol_r) or np.isnan(ch):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        pivot_val = pivot_aligned[i]
        r3 = r3_aligned[i]
        r4 = r4_aligned[i]
        s3 = s3_aligned[i]
        s4 = s4_aligned[i]
        
        if np.isnan(pivot_val) or np.isnan(r3) or np.isnan(r4) or np.isnan(s3) or np.isnan(s4):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit conditions: mean reversion to pivot or chop > 61.8 (ranging) with reversal signal
            if price >= pivot_val or (ch > 61.8 and price <= s3):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: mean reversion to pivot or chop > 61.8 (ranging) with reversal signal
            if price <= pivot_val or (ch > 61.8 and price >= r3):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if ch < 61.8:  # Trending market - breakout strategy
                if price > r4 and vol_r > 1.5:
                    position = 1
                    signals[i] = 0.25
                elif price < s4 and vol_r > 1.5:
                    position = -1
                    signals[i] = -0.25
            else:  # Ranging market - mean reversion strategy
                if price <= s3 and vol_r > 1.5:
                    position = 1
                    signals[i] = 0.25
                elif price >= r3 and vol_r > 1.5:
                    position = -1
                    signals[i] = -0.25
    
    return signals