#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_VolumeSpike_HTFTrend_Regime
Hypothesis: Use 4h timeframe with Donchian(20) breakout, confirmed by 1d EMA50 trend, volume spike, and chop regime filter.
Long when: price breaks above upper Donchian + 1d EMA50 uptrend + volume > 2.0 * avg volume + chop < 61.8 (trending).
Short when: price breaks below lower Donchian + 1d EMA50 downtrend + volume > 2.0 * avg volume + chop < 61.8.
Exit when: price reverts to midpoint of Donchian channel or opposite Donchian level touched.
Uses discrete 0.30 position size to limit fee drag. Designed for BTC/ETH:
- Works in trending markets via breakout with trend filter
- Volume confirmation reduces false breakouts
- Chop regime filter avoids whipsaws in ranging markets
- Targets ~25-50 trades/year for optimal test generalization.
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
    
    # Donchian(20) channels
    lookback = 20
    upper = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lower = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    midpoint = (upper + lower) / 2
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    # Choppiness Index regime filter (14-period)
    chop_period = 14
    atr = pd.Series(np.maximum(np.maximum(high - low, np.abs(high - np.close)), np.abs(low - np.close))).rolling(window=chop_period, min_periods=chop_period).mean().values if False else None
    # Calculate ATR properly
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0  # first period has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_values = pd.Series(tr).rolling(window=chop_period, min_periods=chop_period).mean().values
    
    # Sum of true ranges over chop_period
    sum_tr = pd.Series(tr).rolling(window=chop_period, min_periods=chop_period).sum().values
    # Max high - min low over chop_period
    max_high = pd.Series(high).rolling(window=chop_period, min_periods=chop_period).max().values
    min_low = pd.Series(low).rolling(window=chop_period, min_periods=chop_period).min().values
    range_max_min = max_high - min_low
    
    # Choppiness Index = 100 * log10(sum_tr / range_max_min) / log10(chop_period)
    # Avoid division by zero
    chop = np.zeros(n)
    mask = range_max_min > 0
    chop[mask] = 100 * np.log10(sum_tr[mask] / range_max_min[mask]) / np.log10(chop_period)
    chop[~mask] = 50  # default middle value when range is zero
    
    # Trending regime: chop < 61.8
    trending_regime = chop < 61.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 20 for Donchian/volume, 50 for 1d EMA, 14 for chop
    start_idx = max(lookback, 50, chop_period)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(midpoint[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_spike[i]) or
            np.isnan(chop[i]) or np.isnan(trending_regime[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        size = 0.30  # Fixed position size
        
        if position == 0:
            # Flat - look for breakout with trend, volume, and regime confirmation
            # Long: break above upper + 1d EMA50 uptrend + volume spike + trending regime
            long_entry = (close_val > upper[i]) and \
                       (ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]) and \
                       volume_spike[i] and \
                       trending_regime[i]
            # Short: break below lower + 1d EMA50 downtrend + volume spike + trending regime
            short_entry = (close_val < lower[i]) and \
                        (ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1]) and \
                        volume_spike[i] and \
                        trending_regime[i]
            
            if long_entry:
                signals[i] = size
                position = 1
            elif short_entry:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit when price reverts to midpoint or touches lower Donchian
            if (close_val < midpoint[i]) or (close_val < lower[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when price reverts to midpoint or touches upper Donchian
            if (close_val > midpoint[i]) or (close_val > upper[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian20_Breakout_VolumeSpike_HTFTrend_Regime"
timeframe = "4h"
leverage = 1.0