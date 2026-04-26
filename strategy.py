#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_RegimeFilter_V2
Hypothesis: 12h Camarilla R1/S1 breakout with 1d EMA50 trend filter and choppiness regime (CHOP 30-60) to catch mean reversion in ranging markets. Volume confirmation (>1.5x median) ensures participation. Uses discrete sizing (0.25) to limit churn. Designed for BTC/ETH in both bull/bear markets by aligning with 1d trend while avoiding strong trends via chop filter. Target: 50-150 trades over 4 years.
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
    
    # Volume confirmation: volume > 1.5x 50-period median
    volume_series = pd.Series(volume)
    vol_median = volume_series.rolling(window=50, min_periods=50).median().values
    volume_confirm = volume > (1.5 * vol_median)
    
    # Load 1d data for HTF trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Choppiness regime filter (14-period) - use 4h data for higher frequency chop reading
    df_4h = get_htf_data(prices, '4h')
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
        
        # Regime filter: CHOP between 30 and 60 (avoid strong trends, favor mean reversion)
        in_regime = (chop_aligned[i] >= 30) & (chop_aligned[i] <= 60)
        
        # Long logic: price breaks above R1 + volume confirmation + bullish 1d trend + regime filter
        if close[i] > r1_12h_aligned[i] and volume_confirm[i] and close[i] > ema_50_1d_aligned[i] and in_regime:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: price breaks below S1 + volume confirmation + bearish 1d trend + regime filter
        elif close[i] < s1_12h_aligned[i] and volume_confirm[i] and close[i] < ema_50_1d_aligned[i] and in_regime:
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

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_RegimeFilter_V2"
timeframe = "12h"
leverage = 1.0