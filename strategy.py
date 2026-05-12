#!/usr/bin/env python3
"""
6h_Supertrend_Volume_Regime
Hypothesis: Supertrend captures trend direction, volume confirms strength, and choppy market filter (CM) avoids whipsaws. Works in bull/bear by following Supertrend direction only in trending regimes (CM < 38.2) and avoiding range-bound markets. Uses 6h Supertrend with 1d CM regime filter for robustness.
"""

name = "6h_Supertrend_Volume_Regime"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for chop filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Supertrend on 6h data
    atr_period = 10
    atr_multiplier = 3.0
    
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR
    atr = np.full(n, np.nan)
    for i in range(atr_period, n):
        atr[i] = np.nanmean(tr[i-atr_period+1:i+1])
    
    # Supertrend calculation
    hl2 = (high + low) / 2
    upper_band = hl2 + (atr_multiplier * atr)
    lower_band = hl2 - (atr_multiplier * atr)
    
    supertrend = np.full(n, np.nan)
    trend = np.full(n, 1)  # 1 for uptrend, -1 for downtrend
    
    supertrend[0] = hl2[0]
    for i in range(1, n):
        if np.isnan(atr[i]):
            supertrend[i] = hl2[i]
            trend[i] = trend[i-1]
            continue
            
        if close[i] > upper_band[i-1]:
            trend[i] = 1
        elif close[i] < lower_band[i-1]:
            trend[i] = -1
        else:
            trend[i] = trend[i-1]
            
        if trend[i] == 1:
            supertrend[i] = max(lower_band[i], supertrend[i-1])
        else:
            supertrend[i] = min(upper_band[i], supertrend[i-1])
    
    # Calculate Choppiness Index on 1d data (CM)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for 1d
    tr1_1d = high_1d[1:] - low_1d[1:]
    tr2_1d = np.abs(high_1d[1:] - close_1d[:-1])
    tr3_1d = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.nan], np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))])
    
    # ATR for 1d (period 14)
    atr_1d = np.full(len(close_1d), np.nan)
    atr_period_1d = 14
    for i in range(atr_period_1d, len(close_1d)):
        atr_1d[i] = np.nanmean(tr_1d[i-atr_period_1d+1:i+1])
    
    # Sum of ATR over 14 periods
    sum_atr_1d = np.full(len(close_1d), np.nan)
    for i in range(atr_period_1d, len(close_1d)):
        sum_atr_1d[i] = np.nansum(atr_1d[i-atr_period_1d+1:i+1])
    
    # Max - Min over 14 periods
    max_high_1d = np.full(len(close_1d), np.nan)
    min_low_1d = np.full(len(close_1d), np.nan)
    for i in range(atr_period_1d, len(close_1d)):
        max_high_1d[i] = np.nanmax(high_1d[i-atr_period_1d+1:i+1])
        min_low_1d[i] = np.nanmin(low_1d[i-atr_period_1d+1:i+1])
    
    # Choppiness Index
    cm_1d = np.full(len(close_1d), np.nan)
    for i in range(atr_period_1d, len(close_1d)):
        if sum_atr_1d[i] > 0 and (max_high_1d[i] - min_low_1d[i]) > 0:
            cm_1d[i] = 100 * np.log10(sum_atr_1d[i] / (max_high_1d[i] - min_low_1d[i])) / np.log10(atr_period_1d)
        else:
            cm_1d[i] = np.nan
    
    # Align CM to 6h timeframe
    cm_1d_aligned = align_htf_to_ltf(prices, df_1d, cm_1d)
    
    # Volume spike: >1.5x 20-period average (6h)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.nanmean(volume[i-20:i])
    
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):  # Start after warmup
        if (np.isnan(supertrend[i]) or np.isnan(trend[i]) or 
            np.isnan(cm_1d_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Only trade in trending regime (CM < 38.2)
        if cm_1d_aligned[i] < 38.2:
            if position == 0:
                # LONG: Supertrend uptrend + volume spike
                if trend[i] == 1 and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                # SHORT: Supertrend downtrend + volume spike
                elif trend[i] == -1 and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            elif position == 1:
                # EXIT LONG: Supertrend turns down
                if trend[i] == -1:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # EXIT SHORT: Supertrend turns up
                if trend[i] == 1:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
        else:
            # In ranging market, stay flat
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
    
    return signals