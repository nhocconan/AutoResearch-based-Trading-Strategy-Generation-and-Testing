#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_VolumeFilter_v1
12h timeframe, Camarilla pivot breakout with volume confirmation and ATR filter.
Designed to work in both bull and bear markets by trading breakouts from key intraday levels.
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d Camarilla pivot levels ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    # R4 = Close + (High-Low) * 1.1/2
    # R3 = Close + (High-Low) * 1.1/4
    # R2 = Close + (High-Low) * 1.1/6
    # R1 = Close + (High-Low) * 1.1/12
    # S1 = Close - (High-Low) * 1.1/12
    # S2 = Close - (High-Low) * 1.1/6
    # S3 = Close - (High-Low) * 1.1/4
    # S4 = Close - (High-Low) * 1.1/2
    camarilla_r1 = np.full_like(close_1d, np.nan)
    camarilla_s1 = np.full_like(close_1d, np.nan)
    
    for i in range(len(close_1d)):
        if not (np.isnan(high_1d[i]) or np.isnan(low_1d[i]) or np.isnan(close_1d[i])):
            range_1d = high_1d[i] - low_1d[i]
            camarilla_r1[i] = close_1d[i] + range_1d * 1.1 / 12
            camarilla_s1[i] = close_1d[i] - range_1d * 1.1 / 12
        else:
            camarilla_r1[i] = np.nan
            camarilla_s1[i] = np.nan
    
    # === 1d ATR for volatility filter ===
    # True Range = max[(high-low), |high-prev_close|, |low-prev_close|]
    tr = np.full_like(close_1d, np.nan)
    for i in range(len(close_1d)):
        if i == 0:
            tr[i] = high_1d[i] - low_1d[i]
        else:
            hl = high_1d[i] - low_1d[i]
            hc = abs(high_1d[i] - close_1d[i-1])
            lc = abs(low_1d[i] - close_1d[i-1])
            tr[i] = max(hl, hc, lc)
    
    # ATR(14)
    atr_14 = np.full_like(close_1d, np.nan)
    if len(tr) >= 14:
        atr_14[13] = np.mean(tr[:14])
        for i in range(14, len(tr)):
            atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
    
    # === Align indicators to 12h timeframe ===
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # === 12h Volume confirmation ===
    # Calculate 20-period average volume
    vol_ma_20 = np.full_like(volume, np.nan)
    for i in range(len(volume)):
        if i >= 19:
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
        elif i > 0:
            vol_ma_20[i] = np.mean(volume[max(0, i-9):i+1])
        else:
            vol_ma_20[i] = volume[0]
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_confirm = volume > vol_ma_20 * 1.5
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(atr_14_aligned[i]) or 
            np.isnan(vol_confirm[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: Price breaks above R1 with volume confirmation and ATR filter
            if (close[i] > camarilla_r1_aligned[i] and 
                vol_confirm[i] and 
                atr_14_aligned[i] > 0):  # Ensure volatility is present
                signals[i] = 0.25
                position = 1
                continue
            # Short: Price breaks below S1 with volume confirmation and ATR filter
            elif (close[i] < camarilla_s1_aligned[i] and 
                  vol_confirm[i] and 
                  atr_14_aligned[i] > 0):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: Price returns below S1 (mean reversion) or opposite breakout
            if close[i] < camarilla_s1_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price returns above R1 (mean reversion) or opposite breakout
            if close[i] > camarilla_r1_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_VolumeFilter_v1"
timeframe = "12h"
leverage = 1.0