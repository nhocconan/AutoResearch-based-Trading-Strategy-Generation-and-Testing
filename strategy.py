#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeSpike
Hypothesis: 4h Camarilla R1/S1 breakout with 12h EMA50 trend filter and volume spike (>2x median) for precise entries. Uses chop regime (CHOP > 50) to avoid trending markets where breakouts fail. Discrete position sizing (0.25) minimizes fee churn. Target: 20-50 trades/year. Works in bull/bear by following 12h trend and avoiding false breakouts via chop filter.
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
    
    # Calculate Camarilla levels for 4h (based on previous 4h bar)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    h_4h = df_4h['high'].values
    l_4h = df_4h['low'].values
    c_4h = df_4h['close'].values
    
    # Previous bar's values for level calculation (to avoid look-ahead)
    h_4h_prev = np.roll(h_4h, 1)
    l_4h_prev = np.roll(l_4h, 1)
    c_4h_prev = np.roll(c_4h, 1)
    h_4h_prev[0] = np.nan
    l_4h_prev[0] = np.nan
    c_4h_prev[0] = np.nan
    
    # Calculate Camarilla R1 and S1 levels
    rng_4h = h_4h_prev - l_4h_prev
    r1_4h = c_4h_prev + (rng_4h * 1.1 / 12)
    s1_4h = c_4h_prev - (rng_4h * 1.1 / 12)
    
    # Align to 4h primary timeframe
    r1_4h_aligned = align_htf_to_ltf(prices, df_4h, r1_4h)
    s1_4h_aligned = align_htf_to_ltf(prices, df_4h, s1_4h)
    
    # Volume confirmation: volume > 2x 50-period median
    volume_series = pd.Series(volume)
    vol_median = volume_series.rolling(window=50, min_periods=50).median().values
    volume_confirm = volume > (2.0 * vol_median)
    
    # Load 12h data for HTF trend filter (EMA50)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Choppiness regime filter (14-period) - use 4h data for chop reading
    if len(df_4h) < 20:
        chop_aligned = np.full(n, 50.0)  # neutral default
    else:
        high_4h = df_4h['high'].values
        low_4h = df_4h['low'].values
        close_4h = df_4h['close'].values
        tr = np.maximum(high_4h[1:] - low_4h[1:], np.maximum(np.abs(high_4h[1:] - close_4h[:-1]), np.abs(low_4h[1:] - close_4h[:-1])))
        tr = np.concatenate([[np.nan], tr])
        atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
        highest_high_14 = pd.Series(high_4h).rolling(window=14, min_periods=14).max().values
        lowest_low_14 = pd.Series(low_4h).rolling(window=14, min_periods=14).min().values
        chop = 100 * np.log10(atr_14 * 14 / (highest_high_14 - lowest_low_14)) / np.log10(14)
        chop_aligned = align_htf_to_ltf(prices, df_4h, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 50-period volume median, 50-period EMA, 14-period chop)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_4h_aligned[i]) or np.isnan(s1_4h_aligned[i]) or 
            np.isnan(vol_median[i]) or np.isnan(ema_50_12h_aligned[i]) or np.isnan(chop_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Only trade in choppy markets (CHOP > 50 indicates ranging/mean reverting conditions)
        in_choppy_regime = chop_aligned[i] > 50
        
        # Long logic: price breaks above R1 + volume confirmation + bullish 12h trend + choppy regime
        if close[i] > r1_4h_aligned[i] and volume_confirm[i] and close[i] > ema_50_12h_aligned[i] and in_choppy_regime:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: price breaks below S1 + volume confirmation + bearish 12h trend + choppy regime
        elif close[i] < s1_4h_aligned[i] and volume_confirm[i] and close[i] < ema_50_12h_aligned[i] and in_choppy_regime:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit: opposite breakout (price returns to the other level)
        elif position == 1 and close[i] < s1_4h_aligned[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > r1_4h_aligned[i]:
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

name = "4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0