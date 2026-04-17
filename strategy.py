#!/usr/bin/env python3
"""
12h_Pivot_R1_S1_Breakout_VolumeRegime_v1
Daily Camarilla pivot breakout with volume spike and choppiness regime filter.
Pivot levels calculated from daily OHLC, long/short on break of R1/S1 with volume confirmation.
Choppiness index filter: trade only when CHOP > 61.8 (ranging market) to avoid whipsaws in trends.
Designed for 12h timeframe to target 50-150 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d Camarilla Pivot Levels ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot and levels
    pivot = (high_1d + low_1d + close_1d) / 3
    range_ = high_1d - low_1d
    r1 = close_1d + (range_ * 1.1 / 12)
    s1 = close_1d - (range_ * 1.1 / 12)
    
    # Align to 12h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # === 12h Choppiness Index (14-period) ===
    def true_range(high, low, prev_close):
        tr1 = high - low
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = true_range(high, low, prev_close)
    
    atr14 = np.full_like(close, np.nan)
    for i in range(len(close)):
        if i >= 14:
            if i == 14:
                atr14[i] = np.mean(tr[1:15])
            else:
                atr14[i] = (atr14[i-1] * 13 + tr[i]) / 14
    
    # Sum of true ranges over 14 periods
    sum_tr14 = np.full_like(close, np.nan)
    for i in range(len(close)):
        if i >= 14:
            sum_tr14[i] = np.sum(tr[i-13:i+1])
    
    chop = np.full_like(close, np.nan)
    for i in range(len(close)):
        if i >= 14 and sum_tr14[i] > 0:
            chop[i] = 100 * np.log10(sum_tr14[i] / (atr14[i] * 14)) / np.log10(2)
    
    # Chop > 61.8 indicates ranging market (good for mean reversion)
    chop_filter = chop > 61.8
    
    # === 12h Volume confirmation (20-period average) ===
    vol_ma_20 = np.full_like(volume, np.nan)
    for i in range(len(volume)):
        if i >= 20:
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    vol_confirm = volume > vol_ma_20 * 1.5  # volume spike: 1.5x average
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(chop_filter[i]) or 
            np.isnan(vol_confirm[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price breaks above R1 with volume confirmation in ranging market
            if (close[i] > r1_aligned[i] and 
                vol_confirm[i] and 
                chop_filter[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below S1 with volume confirmation in ranging market
            elif (close[i] < s1_aligned[i] and 
                  vol_confirm[i] and 
                  chop_filter[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price returns to pivot or opposite break with volume
            if (close[i] < pivot_aligned[i] or 
                (close[i] < s1_aligned[i] and vol_confirm[i])):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to pivot or opposite break with volume
            if (close[i] > pivot_aligned[i] or 
                (close[i] > r1_aligned[i] and vol_confirm[i])):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Pivot_R1_S1_Breakout_VolumeRegime_v1"
timeframe = "12h"
leverage = 1.0