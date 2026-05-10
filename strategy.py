#!/usr/bin/env python3
# 12h_TRIX_Trend_With_Volume_Spike_and_Chop_Filter
# Hypothesis: TRIX momentum combined with volume spike and chop regime filter on 12h timeframe.
# TRIX filters out noise and identifies smooth momentum. Volume spike confirms institutional participation.
# Chop filter avoids whipsaws in ranging markets. Designed for low trade frequency and high edge.
# Uses 1d timeframe for chop regime (20-period) and volume confirmation (20-period MA).
# Expected trade count: 15-25 per year per symbol (60-100 total over 4 years).

name = "12h_TRIX_Trend_With_Volume_Spike_and_Chop_Filter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for chop regime and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate TRIX on 12h close (15-period EMA triple)
    # TRIX = EMA(EMA(EMA(close, 15), 15), 15) - 1
    close_series = pd.Series(close)
    ema1 = close_series.ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = ema2.ewm(span=15, adjust=False, min_periods=15).mean()
    trix = (ema3 / ema3.shift(1) - 1) * 100
    trix = trix.fillna(0).values
    
    # Calculate chop regime on 1d (high-low range / ATR-like measure)
    # Chop = 100 * log10(sum(high-low, 14) / (max(high,14) - min(low,14))) / log10(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True range components for chop calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = 0  # first value has no previous close
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_tr_14 / (max_high_14 - min_low_14)) / np.log10(14)
    chop = np.where((max_high_14 - min_low_14) == 0, 50, chop)  # avoid division by zero
    
    # Volume confirmation: 20-period MA on 1d volume
    volume_1d = df_1d['volume'].values
    volume_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 12h timeframe
    trix_12h = trix  # TRIX is already on 12h
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need TRIX (45 for triple EMA), chop (14), volume MA (20)
    start_idx = max(45, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(trix_12h[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(volume_ma_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # TRIX momentum signal
        trix_rising = trix_12h[i] > trix_12h[i-1]
        trix_falling = trix_12h[i] < trix_12h[i-1]
        
        # Chop regime: < 38.2 = trending, > 61.8 = ranging
        chopping = chop_aligned[i] > 61.8
        trending = chop_aligned[i] < 38.2
        
        # Volume confirmation: current 12h volume > 20-period 1d volume MA
        # Note: comparing 12h volume to 1d volume MA requires scaling approximation
        # Using volume ratio: current volume > 1.5 * aligned volume MA
        volume_confirm = volume[i] > volume_ma_aligned[i] * 1.5
        
        if position == 0:
            # Long entry: TRIX rising + trending regime + volume spike
            if trix_rising and trending and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: TRIX falling + trending regime + volume spike
            elif trix_falling and trending and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: TRIX falling OR choppy regime OR volume dies
            if trix_falling or chopping or not volume_confirm:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: TRIX rising OR choppy regime OR volume dies
            if trix_rising or chopping or not volume_confirm:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals