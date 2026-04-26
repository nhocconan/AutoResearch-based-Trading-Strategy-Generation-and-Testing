#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_WeeklyTrend_VolumeFilter
Hypothesis: Daily Camarilla R1/S1 breakout with weekly EMA50 trend filter and volume spike (>2x median).
Only enters in ranging markets (CHOP(14) > 61.8) to avoid whipsaws in strong trends.
Goes long when price breaks above R1 with volume spike, weekly trend bullish (price > EMA50), and choppy regime.
Goes short when price breaks below S1 with volume spike, weekly trend bearish (price < EMA50), and choppy regime.
Uses discrete position sizing (0.25) to minimize churn. Designed for 30-100 total trades over 4 years.
Works in both bull and bear markets by following weekly trend filter and avoiding strong trends via chop filter.
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
    
    # Calculate Camarilla levels for 1d (based on previous 1d bar)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    # Previous bar's values for level calculation (to avoid look-ahead)
    h_1d_prev = np.roll(h_1d, 1)
    l_1d_prev = np.roll(l_1d, 1)
    c_1d_prev = np.roll(c_1d, 1)
    h_1d_prev[0] = np.nan
    l_1d_prev[0] = np.nan
    c_1d_prev[0] = np.nan
    
    # Calculate Camarilla R1 and S1 levels
    rng_1d = h_1d_prev - l_1d_prev
    r1_1d = c_1d_prev + (rng_1d * 1.1 / 6)
    s1_1d = c_1d_prev - (rng_1d * 1.1 / 6)
    
    # Align to 1d primary timeframe
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Volume spike: volume > 2x 50-period median
    volume_series = pd.Series(volume)
    vol_median = volume_series.rolling(window=50, min_periods=50).median().values
    volume_spike = volume > (2.0 * vol_median)
    
    # Load weekly data for HTF trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Choppiness regime filter: CHOP(14) > 61.8 = ranging market (avoid strong trends)
    # True range = max(high-low, abs(high-previous_close), abs(low-previous_close))
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    # Sum of true ranges over 14 periods
    sum_tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Choppiness index: 100 * log10(sum_tr14 / (atr14 * 14)) / log10(14)
    chop = 100 * np.log10(sum_tr14 / (atr14 * 14)) / np.log10(14)
    chop[np.isnan(chop) | (atr14 == 0)] = 50  # default to neutral when undefined
    chop_regime = chop > 61.8  # ranging market
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 50-period volume median, 50-period EMA, 14-period chop)
    start_idx = max(50, 50, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or 
            np.isnan(vol_median[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(chop_regime[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long logic: price breaks above R1 + volume spike + bullish weekly trend + choppy regime
        if close[i] > r1_1d_aligned[i] and volume_spike[i] and close[i] > ema_50_1w_aligned[i] and chop_regime[i]:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: price breaks below S1 + volume spike + bearish weekly trend + choppy regime
        elif close[i] < s1_1d_aligned[i] and volume_spike[i] and close[i] < ema_50_1w_aligned[i] and chop_regime[i]:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit: opposite breakout (price returns to median levels)
        elif position == 1 and close[i] < (r1_1d_aligned[i] + s1_1d_aligned[i]) / 2:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > (r1_1d_aligned[i] + s1_1d_aligned[i]) / 2:
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

name = "1d_Camarilla_R1_S1_Breakout_WeeklyTrend_VolumeFilter"
timeframe = "1d"
leverage = 1.0