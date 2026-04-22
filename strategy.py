#!/usr/bin/env python3
"""
12-hour Camarilla Pivot Reversal with 1-day Volume Spike and Chop Filter
Long when price crosses above S3 with volume spike and chop > 61.8 (range)
Short when price crosses below R3 with volume spike and chop > 61.8 (range)
Exit when price crosses S2/R2 or chop < 38.2 (trend)
Camarilla levels provide reversal points in range markets; volume spike confirms institutional interest.
Chop filter ensures we only trade in ranging conditions, avoiding whipsaws in trends.
Works in both bull and bear markets by exploiting mean reversion in ranges.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1-day data for Camarilla, volume, and chop - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    # Camarilla: Based on previous day's high, low, close
    phigh = df_1d['high'].shift(1).values  # Previous day high
    plow = df_1d['low'].shift(1).values    # Previous day low
    pclose = df_1d['close'].shift(1).values # Previous day close
    
    range_ = phigh - plow
    # Camarilla levels
    s3 = pclose - (range_ * 1.1 / 4)
    s2 = pclose - (range_ * 1.1 / 6)
    s1 = pclose - (range_ * 1.1 / 12)
    r1 = pclose + (range_ * 1.1 / 12)
    r2 = pclose + (range_ * 1.1 / 6)
    r3 = pclose + (range_ * 1.1 / 4)
    
    # Align Camarilla levels to 12h timeframe
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    
    # Volume spike: current 1-day volume > 20-period average
    volume_1d = df_1d['volume'].values
    avg_vol_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > avg_vol_1d
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike.astype(float))
    
    # Chopiness Index (14-period) for regime detection
    # Chop = 100 * log10(sum(atr1) / (max(high) - min(low))) / log10(14)
    tr1 = np.maximum(df_1d['high'].values - df_1d['low'].values,
                     np.maximum(abs(df_1d['high'].values - df_1d['close'].shift(1).values),
                                abs(df_1d['low'].values - df_1d['close'].shift(1).values)))
    atr1 = pd.Series(tr1).rolling(window=1, min_periods=1).sum().values  # True Range
    sum_atr1 = pd.Series(atr1).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    range_14 = max_high - min_low
    chop = 100 * (np.log10(sum_atr1 / range_14) / np.log10(14))
    chop = np.where(range_14 == 0, 50, chop)  # Avoid division by zero
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(s3_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(vol_spike_aligned[i]) or np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price crosses above S3 with volume spike and chop > 61.8 (range)
            if (close[i] > s3_aligned[i] and close[i-1] <= s3_aligned[i-1] and
                vol_spike_aligned[i] > 0.5 and chop_aligned[i] > 61.8):
                signals[i] = 0.25
                position = 1
            # Short: Price crosses below R3 with volume spike and chop > 61.8 (range)
            elif (close[i] < r3_aligned[i] and close[i-1] >= r3_aligned[i-1] and
                  vol_spike_aligned[i] > 0.5 and chop_aligned[i] > 61.8):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price crosses above S2 or chop < 38.2 (trend)
                if (close[i] > s2_aligned[i] and close[i-1] <= s2_aligned[i-1]) or \
                   chop_aligned[i] < 38.2:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price crosses below R2 or chop < 38.2 (trend)
                if (close[i] < r2_aligned[i] and close[i-1] >= r2_aligned[i-1]) or \
                   chop_aligned[i] < 38.2:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Camarilla_S3R3_VolumeSpike_ChopFilter"
timeframe = "12h"
leverage = 1.0