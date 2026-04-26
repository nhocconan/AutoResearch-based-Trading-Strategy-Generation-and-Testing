#!/usr/bin/env python3
"""
4h_TRIX_VolumeSpike_ChopRegime_v1
Hypothesis: Trade 4h TRIX zero-cross with volume spike and choppiness regime filter.
Long when TRIX crosses above zero AND volume spike AND CHOP > 61.8 (range).
Short when TRIX crosses below zero AND volume spike AND CHOP > 61.8 (range).
Exit on opposite TRIX cross. Uses 1d trend filter to avoid counter-trend in strong trends.
Designed for low trade frequency (<50/year) with high edge in ranging/transition markets.
Works in bull via range longs in accumulation, in bear via range shorts in distribution.
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
    
    # Get 1d data for trend filter and choppiness
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA(50) on 1d for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate choppiness index on 1d (CHOP > 61.8 = ranging)
    # CHOP = 100 * log10(sum(ATR(14)) / log10(range(14))) / log10(14)
    atr_period = 14
    chop_period = 14
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    # ATR(14)
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Sum of ATR over chop_period
    sum_atr = pd.Series(atr).rolling(window=chop_period, min_periods=chop_period).sum().values
    
    # Max(high) and min(low) over chop_period
    max_high = pd.Series(high_1d).rolling(window=chop_period, min_periods=chop_period).max().values
    min_low = pd.Series(low_1d).rolling(window=chop_period, min_periods=chop_period).min().values
    range_chop = max_high - min_low
    
    # Choppiness Index
    chop = np.zeros_like(close_1d)
    mask = (sum_atr > 0) & (range_chop > 0)
    chop[mask] = 100 * np.log10(sum_atr[mask] / range_chop[mask]) / np.log10(chop_period)
    chop = np.where(chop == 0, 50, chop)  # default to neutral
    
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate TRIX on 4h (primary timeframe)
    # TRIX = EMA(EMA(EMA(close, 12), 12), 12) - 1 period ago, then percent change
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    trix = pd.Series(ema3).pct_change(periods=1) * 100  # percent change
    trix_values = trix.values
    trix_values = np.where(np.isnan(trix_values), 0, trix_values)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of 1d EMA(50), chop, TRIX(36), volume MA(20)
    start_idx = max(50, chop_period, 36, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(chop_aligned[i]) or
            np.isnan(trix_values[i]) or
            np.isnan(vol_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        close_val = close[i]
        vol_conf = volume_confirm[i]
        chop_val = chop_aligned[i]
        trix_now = trix_values[i]
        trix_prev = trix_values[i-1]
        trend_up = close_val > ema_50_1d_aligned[i]   # 1d uptrend
        trend_down = close_val < ema_50_1d_aligned[i]  # 1d downtrend
        
        # Regime filter: only trade in ranging markets (CHOP > 61.8)
        in_range = chop_val > 61.8
        
        if position == 0:
            # Long: TRIX crosses above zero AND volume confirm AND in range
            long_signal = (trix_prev <= 0) and (trix_now > 0) and vol_conf and in_range
            
            # Short: TRIX crosses below zero AND volume confirm AND in range
            short_signal = (trix_prev >= 0) and (trix_now < 0) and vol_conf and in_range
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: TRIX crosses below zero OR 1d trend flips down (avoid counter-trend)
            if (trix_prev >= 0 and trix_now < 0) or (not trend_up):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: TRIX crosses above zero OR 1d trend flips up (avoid counter-trend)
            if (trix_prev <= 0 and trix_now > 0) or (not trend_down):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_TRIX_VolumeSpike_ChopRegime_v1"
timeframe = "4h"
leverage = 1.0