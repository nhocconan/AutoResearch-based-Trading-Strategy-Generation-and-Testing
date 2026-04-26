#!/usr/bin/env python3
"""
6h_Camarilla_R4_S4_Breakout_12hTrend_VolumeRegime
Hypothesis: 6h Camarilla R4/S4 breakout with 12h EMA50 trend filter and volume spike (>2x median).
R4/S4 levels represent stronger breakout points than R3/S3, reducing false signals.
Only enters in low-volatility regime (Choppiness Index > 61.8) to avoid whipsaws in strong trends.
Uses discrete position sizing (0.25) to minimize churn. Target: 50-150 total trades over 4 years.
Works in both bull and bear markets by following 12h trend filter and avoiding strong trends via chop filter.
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
    
    # Calculate Camarilla levels for 6h (based on previous 6h bar)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 2:
        return np.zeros(n)
    
    h_6h = df_6h['high'].values
    l_6h = df_6h['low'].values
    c_6h = df_6h['close'].values
    
    # Previous bar's values for level calculation (to avoid look-ahead)
    h_6h_prev = np.roll(h_6h, 1)
    l_6h_prev = np.roll(l_6h, 1)
    c_6h_prev = np.roll(c_6h, 1)
    h_6h_prev[0] = np.nan
    l_6h_prev[0] = np.nan
    c_6h_prev[0] = np.nan
    
    # Calculate Camarilla R4 and S4 levels (stronger breakout points)
    rng_6h = h_6h_prev - l_6h_prev
    r4_6h = c_6h_prev + (rng_6h * 1.1 / 2)
    s4_6h = c_6h_prev - (rng_6h * 1.1 / 2)
    
    # Align to 6h primary timeframe
    r4_6h_aligned = align_htf_to_ltf(prices, df_6h, r4_6h)
    s4_6h_aligned = align_htf_to_ltf(prices, df_6h, s4_6h)
    
    # Volume spike: volume > 2x 50-period median
    volume_series = pd.Series(volume)
    vol_median = volume_series.rolling(window=50, min_periods=50).median().values
    volume_spike = volume > (2.0 * vol_median)
    
    # Load 12h data for HTF trend filter (EMA50)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
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
        if (np.isnan(r4_6h_aligned[i]) or np.isnan(s4_6h_aligned[i]) or 
            np.isnan(vol_median[i]) or np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(chop_regime[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long logic: price breaks above R4 + volume spike + bullish 12h trend + choppy regime
        if close[i] > r4_6h_aligned[i] and volume_spike[i] and close[i] > ema_50_12h_aligned[i] and chop_regime[i]:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: price breaks below S4 + volume spike + bearish 12h trend + choppy regime
        elif close[i] < s4_6h_aligned[i] and volume_spike[i] and close[i] < ema_50_12h_aligned[i] and chop_regime[i]:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit: opposite breakout (price returns to median levels)
        elif position == 1 and close[i] < (r4_6h_aligned[i] + s4_6h_aligned[i]) / 2:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > (r4_6h_aligned[i] + s4_6h_aligned[i]) / 2:
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

name = "6h_Camarilla_R4_S4_Breakout_12hTrend_VolumeRegime"
timeframe = "6h"
leverage = 1.0