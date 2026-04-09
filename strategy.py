#!/usr/bin/env python3
# 12h_camarilla_pivot_volume_regime_v1
# Hypothesis: 12h strategy using Camarilla pivot levels from 1d timeframe with volume confirmation and chop regime filter.
# Long: Price touches S3 support, volume > 1.5x 20-period average, CHOP > 61.8 (ranging market).
# Short: Price touches R3 resistance, volume > 1.5x 20-period average, CHOP > 61.8 (ranging market).
# Exit: Opposite pivot touch or CHOP < 38.2 (trending market).
# Uses 1d Camarilla levels for higher timeframe structure and 1d CHOP for regime detection.
# Volume confirmation filters weak breakouts. CHOP filter ensures mean-reversion logic only in ranging markets.
# Target: 12-37 trades/year (50-150 total over 4 years).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_volume_regime_v1"
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Camarilla pivot levels
    # Camarilla: Based on previous day's OHLC
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Typical price for pivot calculation
    typical_price_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels (based on previous day)
    # S3 = close - (high - low) * 1.1/4
    # S2 = close - (high - low) * 1.1/6
    # S1 = close - (high - low) * 1.1/12
    # R1 = close + (high - low) * 1.1/12
    # R2 = close + (high - low) * 1.1/6
    # R3 = close + (high - low) * 1.1/4
    s3_1d = close_1d - (range_1d * 1.1 / 4)
    s2_1d = close_1d - (range_1d * 1.1 / 6)
    s1_1d = close_1d - (range_1d * 1.1 / 12)
    r1_1d = close_1d + (range_1d * 1.1 / 12)
    r2_1d = close_1d + (range_1d * 1.1 / 6)
    r3_1d = close_1d + (range_1d * 1.1 / 4)
    
    # Align 1d Camarilla levels to 12h timeframe (completed 1d bar only)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s2_1d_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    r2_1d_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    
    # Calculate 1d Choppiness Index (CHOP) for regime filter
    # CHOP = 100 * log10(sum(ATR(14)) / log10(n) * (highest_high - lowest_low))
    # Simplified: CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    atr_period = 14
    high_low_1d = np.maximum(high_1d, np.concatenate([[high_1d[0]], high_1d[:-1]])) - np.minimum(low_1d, np.concatenate([[low_1d[0]], low_1d[:-1]]))
    high_low_1d = np.maximum(high_1d - low_1d, high_low_1d)
    atr_1d = pd.Series(high_low_1d).rolling(window=atr_period, min_periods=atr_period).mean().values
    highest_high_1d = pd.Series(high_1d).rolling(window=atr_period, min_periods=atr_period).max().values
    lowest_low_1d = pd.Series(low_1d).rolling(window=atr_period, min_periods=atr_period).min().values
    
    # Avoid division by zero
    true_range_sum = atr_1d * atr_period
    price_range = highest_high_1d - lowest_low_1d
    chop_1d = np.where(price_range > 0, 100 * np.log10(true_range_sum / price_range) / np.log10(atr_period), 50)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Volume average for confirmation (20-period on 12h)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(s3_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or
            np.isnan(chop_1d_aligned[i]) or np.isnan(volume_ma[i]) or
            np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        # CHOP regime filter: only trade in ranging markets (CHOP > 61.8)
        chop_filter = chop_1d_aligned[i] > 61.8
        
        if position == 1:  # Long position
            # Exit: Price touches R3 (opposite pivot) OR CHOP < 38.2 (trending market)
            if close[i] >= r3_1d_aligned[i] or chop_1d_aligned[i] < 38.2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price touches S3 (opposite pivot) OR CHOP < 38.2 (trending market)
            if close[i] <= s3_1d_aligned[i] or chop_1d_aligned[i] < 38.2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: Price touches S3 support, volume confirmed, chop filter (ranging)
            if (close[i] <= s3_1d_aligned[i] and volume_confirmed and chop_filter):
                position = 1
                signals[i] = 0.25
            # Short entry: Price touches R3 resistance, volume confirmed, chop filter (ranging)
            elif (close[i] >= r3_1d_aligned[i] and volume_confirmed and chop_filter):
                position = -1
                signals[i] = -0.25
    
    return signals