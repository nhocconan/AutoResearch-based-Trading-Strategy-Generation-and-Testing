#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1wTrend_VolumeRegime
Hypothesis: 12h Camarilla R1/S1 breakout with 1w trend filter (price > EMA50) and volume confirmation (>1.5x EMA20 volume) in low-chop regime (CHOP < 50).
Enters long when price breaks above R1 with bullish 1w trend, volume spike, and low chop.
Enters short when price breaks below S1 with bearish 1w trend, volume spike, and low chop.
Exits when price reverts to opposite Camarilla level (S1 for longs, R1 for shorts).
Designed for 50-150 total trades over 4 years (12-37/year) to avoid fee drag.
Uses discrete position sizing (0.25) to minimize churn. Works in both bull and bear markets by following 1w trend.
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
    
    # Calculate Camarilla pivot levels (R1, S1) on 12h timeframe
    # Use prior completed 12h bar to avoid look-ahead
    df_12h = get_htf_data(prices, '12h')
    
    # Prior 12h bar's OHLC for Camarilla calculation (shifted by 1)
    prior_high = np.roll(df_12h['high'].values, 1)
    prior_low = np.roll(df_12h['low'].values, 1)
    prior_close = np.roll(df_12h['close'].values, 1)
    prior_open = np.roll(df_12h['open'].values, 1)
    prior_high[0] = np.nan
    prior_low[0] = np.nan
    prior_close[0] = np.nan
    prior_open[0] = np.nan
    
    # Calculate pivot point and Camarilla levels (R1, S1)
    pivot = (prior_high + prior_low + prior_close) / 3.0
    range_hl = prior_high - prior_low
    r1 = pivot + range_hl * 1.1 / 4.0
    s1 = pivot - range_hl * 1.1 / 4.0
    
    # Align Camarilla levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_12h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_12h, s1)
    
    # Load 1w data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: volume > 1.5 * 20-period EMA volume
    avg_volume = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * avg_volume)
    
    # Choppiness regime filter on 1d timeframe (CHOP < 50 = low chop/trending)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with 1d index
    
    # ATR(14)
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: CHOP = 100 * log10(sum(TR14) / (HH14 - LL14)) / log10(14)
    sum_tr14 = pd.Series(atr_14 * 14).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_tr14 / (hh_14 - ll_14)) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Low chop regime: CHOP < 50
    low_chop = chop_aligned < 50
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 12h shift + 50-period EMA + 14-period chop)
    start_idx = 1 + 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(low_chop[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long logic: break above R1 + bullish 1w trend + volume spike + low chop
        if close[i] > r1_aligned[i] and close[i] > ema_50_1w_aligned[i] and volume_spike[i] and low_chop[i]:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: break below S1 + bearish 1w trend + volume spike + low chop
        elif close[i] < s1_aligned[i] and close[i] < ema_50_1w_aligned[i] and volume_spike[i] and low_chop[i]:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit: price reverts to opposite Camarilla level
        elif position == 1 and close[i] < s1_aligned[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > r1_aligned[i]:
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

name = "12h_Camarilla_R1_S1_Breakout_1wTrend_VolumeRegime"
timeframe = "12h"
leverage = 1.0