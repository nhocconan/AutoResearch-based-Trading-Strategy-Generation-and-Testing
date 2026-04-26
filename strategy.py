#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_Regime
Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter, volume spike (>2x median), and choppiness regime filter.
Only enters when CHOP(14) > 61.8 (ranging market) to avoid whipsaws in strong trends.
Goes long when price breaks above R3 with volume spike, 1d trend bullish (price > EMA34), and choppy regime.
Goes short when price breaks below S3 with volume spike, 1d trend bearish (price < EMA34), and choppy regime.
Uses discrete position sizing (0.25) to minimize churn. Designed for 75-200 total trades over 4 years.
Works in both bull and bear markets by following 1d trend filter and avoiding strong trends via chop filter.
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
    
    # Calculate Camarilla R3 and S3 levels
    rng_4h = h_4h_prev - l_4h_prev
    r3_4h = c_4h_prev + (rng_4h * 1.1 / 4)
    s3_4h = c_4h_prev - (rng_4h * 1.1 / 4)
    
    # Align to 4h primary timeframe
    r3_4h_aligned = align_htf_to_ltf(prices, df_4h, r3_4h)
    s3_4h_aligned = align_htf_to_ltf(prices, df_4h, s3_4h)
    
    # Volume spike: volume > 2x 50-period median
    volume_series = pd.Series(volume)
    vol_median = volume_series.rolling(window=50, min_periods=50).median().values
    volume_spike = volume > (2.0 * vol_median)
    
    # Load 1d data for HTF trend filter (EMA34)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
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
    
    # Start after warmup (need 50-period volume median, 34-period EMA, 14-period chop)
    start_idx = max(50, 34, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r3_4h_aligned[i]) or np.isnan(s3_4h_aligned[i]) or 
            np.isnan(vol_median[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(chop_regime[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long logic: price breaks above R3 + volume spike + bullish 1d trend + choppy regime
        if close[i] > r3_4h_aligned[i] and volume_spike[i] and close[i] > ema_34_1d_aligned[i] and chop_regime[i]:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: price breaks below S3 + volume spike + bearish 1d trend + choppy regime
        elif close[i] < s3_4h_aligned[i] and volume_spike[i] and close[i] < ema_34_1d_aligned[i] and chop_regime[i]:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit: opposite breakout (price returns to median levels)
        elif position == 1 and close[i] < (r3_4h_aligned[i] + s3_4h_aligned[i]) / 2:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > (r3_4h_aligned[i] + s3_4h_aligned[i]) / 2:
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

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_Regime"
timeframe = "4h"
leverage = 1.0