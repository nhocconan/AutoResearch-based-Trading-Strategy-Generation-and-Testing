#!/usr/bin/env python3
"""
4h_1d_Camarilla_Breakout_Volume_Regime_v1
Hypothesis: On 4h timeframe, enter long when price breaks above daily Camarilla R3 with volume confirmation (volume > 1.5x 20-period average) and chop regime filter (Choppiness Index > 61.8 for mean reversion regime). Enter short when price breaks below daily Camarilla S3 with same filters. Uses daily Camarilla levels for structure, volume for conviction, and chop regime to avoid trending markets where breakouts fail. Designed for ~25-40 trades/year to avoid fee drag and work in both bull (breakouts) and bear (mean reversion in chop) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Camarilla_Breakout_Volume_Regime_v1"
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
    
    # === DAILY INDICATORS: Camarilla pivot levels ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily Camarilla pivot levels
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla R3 and S3 levels (key reversal levels)
    r3 = close_1d + range_1d * 1.1 / 4
    s3 = close_1d - range_1d * 1.1 / 4
    
    # Align to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # === VOLUME INDICATOR: 20-period average volume ===
    vol_ma = np.full(n, np.nan)
    if n >= 20:
        vol_ma[19] = np.mean(volume[:20])
        for i in range(20, n):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full(n, np.nan)
    for i in range(n):
        if vol_ma[i] > 0:
            volume_ratio[i] = volume[i] / vol_ma[i]
    
    # === CHOPPINESS INDEX (14-period) for regime detection ===
    def true_range(high, low, close_prev):
        return np.maximum(high - low, np.maximum(np.abs(high - close_prev), np.abs(low - close_prev)))
    
    tr = np.full(n, np.nan)
    if n >= 1:
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = true_range(high[i], low[i], close[i-1])
    
    # Sum of TR over 14 periods
    tr_sum14 = np.full(n, np.nan)
    if n >= 14:
        tr_sum14[13] = np.sum(tr[:14])
        for i in range(14, n):
            tr_sum14[i] = tr_sum14[i-1] - tr[i-14] + tr[i]
    
    # Calculate highest high and lowest low over 14 periods
    max_high14 = np.full(n, np.nan)
    min_low14 = np.full(n, np.nan)
    if n >= 14:
        max_high14[13] = np.max(high[:14])
        min_low14[13] = np.min(low[:14])
        for i in range(14, n):
            max_high14[i] = max(max_high14[i-1], high[i])
            min_low14[i] = min(min_low14[i-1], low[i])
    
    # Chop = 100 * log10(sum(tr14) / (max_high14 - min_low14)) / log10(14)
    chop = np.full(n, np.nan)
    for i in range(14, n):
        if max_high14[i] > min_low14[i]:
            chop[i] = 100 * np.log10(tr_sum14[i] / (max_high14[i] - min_low14[i])) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # start after warmup
        # Skip if indicators not available
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(chop[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: volume > 1.5x average
        volume_confirm = volume_ratio[i] > 1.5
        
        # Chop regime: > 61.8 = ranging market (good for mean reversion at extremes)
        chop_regime = chop[i] > 61.8
        
        # Breakout conditions with filters
        long_breakout = (close[i] > r3_aligned[i]) and volume_confirm and chop_regime
        short_breakout = (close[i] < s3_aligned[i]) and volume_confirm and chop_regime
        
        # Exit conditions: reversal back to opposite extreme or volume/chop deterioration
        # Exit long if price returns to S3 or volume/chop deteriorates
        exit_long = (close[i] < s3_aligned[i]) or (volume_ratio[i] < 1.2) or (chop[i] < 50)
        # Exit short if price returns to R3 or volume/chop deteriorates
        exit_short = (close[i] > r3_aligned[i]) or (volume_ratio[i] < 1.2) or (chop[i] < 50)
        
        if long_breakout and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_breakout and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals