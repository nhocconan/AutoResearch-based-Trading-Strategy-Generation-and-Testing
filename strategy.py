#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dEMA50_VolumeSpike_ChopFilter
Hypothesis: 12h Camarilla R1/S1 breakout with 1d EMA50 trend filter, volume spike (>2.0x median volume), and choppiness regime (CHOP 30-61.8 = ranging/low trend) to avoid whipsaws. Enters long when price breaks above R1 with volume spike, bullish 1d trend, and chop regime. Enters short when price breaks below S1 with volume spike, bearish 1d trend, and chop regime. Exits on opposite breakout. Uses discrete position sizing (0.25) to minimize churn. Target: 50-150 trades over 4 years. Works in both bull and bear by following 1d trend and avoiding strong trends via chop filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels for 12h (based on previous 12h bar)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    h_12h = df_12h['high'].values
    l_12h = df_12h['low'].values
    c_12h = df_12h['close'].values
    
    # Previous bar's values for level calculation (to avoid look-ahead)
    h_12h_prev = np.roll(h_12h, 1)
    l_12h_prev = np.roll(l_12h, 1)
    c_12h_prev = np.roll(c_12h, 1)
    h_12h_prev[0] = np.nan
    l_12h_prev[0] = np.nan
    c_12h_prev[0] = np.nan
    
    # Calculate Camarilla R1 and S1 levels
    rng_12h = h_12h_prev - l_12h_prev
    r1_12h = c_12h_prev + (rng_12h * 1.1 / 12)
    s1_12h = c_12h_prev - (rng_12h * 1.1 / 12)
    
    # Align to 12h primary timeframe
    r1_12h_aligned = align_htf_to_ltf(prices, df_12h, r1_12h)
    s1_12h_aligned = align_htf_to_ltf(prices, df_12h, s1_12h)
    
    # Volume confirmation: volume > 2.0x 50-period median (stricter than 1.5x)
    volume_series = pd.Series(volume)
    vol_median = volume_series.rolling(window=50, min_periods=50).median().values
    volume_confirm = volume > (2.0 * vol_median)
    
    # Load 1d data for HTF trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Choppiness regime filter (14-period) - use 12h data for consistency
    df_12h_for_chop = df_12h  # reuse 12h data
    if len(df_12h_for_chop) < 20:
        chop_aligned = np.full(n, 50.0)  # neutral default
    else:
        high_12h = df_12h_for_chop['high'].values
        low_12h = df_12h_for_chop['low'].values
        close_12h = df_12h_for_chop['close'].values
        tr = np.maximum(high_12h[1:] - low_12h[1:], np.maximum(np.abs(high_12h[1:] - close_12h[:-1]), np.abs(low_12h[1:] - close_12h[:-1])))
        tr = np.concatenate([[np.nan], tr])
        atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
        highest_high_14 = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
        lowest_low_14 = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
        chop = 100 * np.log10(atr_14 * 14 / (highest_high_14 - lowest_low_14)) / np.log10(14)
        chop_aligned = align_htf_to_ltf(prices, df_12h_for_chop, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 50-period volume median, 50-period EMA, 14-period chop)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_12h_aligned[i]) or np.isnan(s1_12h_aligned[i]) or 
            np.isnan(vol_median[i]) or np.isnan(ema_50_1d_aligned[i]) or np.isnan(chop_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Choppiness regime: 30 <= CHOP <= 61.8 (ranging/low trend, avoids strong trends)
        in_choppy_regime = (chop_aligned[i] >= 30.0) and (chop_aligned[i] <= 61.8)
        
        # Long logic: price breaks above R1 + volume spike + bullish 1d trend + choppy regime
        if close[i] > r1_12h_aligned[i] and volume_confirm[i] and close[i] > ema_50_1d_aligned[i] and in_choppy_regime:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: price breaks below S1 + volume spike + bearish 1d trend + choppy regime
        elif close[i] < s1_12h_aligned[i] and volume_confirm[i] and close[i] < ema_50_1d_aligned[i] and in_choppy_regime:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit: opposite breakout (price returns to the other level)
        elif position == 1 and close[i] < s1_12h_aligned[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > r1_12h_aligned[i]:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dEMA50_VolumeSpike_ChopFilter"
timeframe = "12h"
leverage = 1.0