#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_HTFTrend_VolumeRegime
Hypothesis: Use 4h timeframe with Donchian(20) breakout, confirmed by 1d EMA50 trend and volume regime filter.
Long when: price breaks above Donchian upper + 1d EMA50 uptrend + volume > 1.2 * avg volume.
Short when: price breaks below Donchian lower + 1d EMA50 downtrend + volume > 1.2 * avg volume.
Exit when: price reverts to Donchian midpoint or opposite Donchian level touched.
Uses discrete 0.25 position size to limit fee drag. Targets 20-50 trades/year for optimal test generalization.
Works in both bull and bear markets via trend filter and volume regime to avoid false breakouts.
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
    donchian_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    donchian_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume regime: current volume > 1.2 * 20-period average (avoid low-volume breakouts)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_regime = volume > (1.2 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 20 for Donchian, 50 for 1d EMA
    start_idx = max(lookback, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(donchian_mid[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(volume_regime[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        size = 0.25  # Fixed position size
        
        if position == 0:
            # Flat - look for breakout with trend and volume confirmation
            # Long: break above Donchian high + 1d EMA50 uptrend + volume regime
            long_entry = (close_val > donchian_high[i]) and \
                       (ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]) and \
                       volume_regime[i]
            # Short: break below Donchian low + 1d EMA50 downtrend + volume regime
            short_entry = (close_val < donchian_low[i]) and \
                        (ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1]) and \
                        volume_regime[i]
            
            if long_entry:
                signals[i] = size
                position = 1
            elif short_entry:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit when price reverts to midpoint or touches Donchian low
            if (close_val < donchian_mid[i]) or (close_val < donchian_low[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when price reverts to midpoint or touches Donchian high
            if (close_val > donchian_mid[i]) or (close_val > donchian_high[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian20_Breakout_HTFTrend_VolumeRegime"
timeframe = "4h"
leverage = 1.0