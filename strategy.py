#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dTrendFilter_VolumeSpike_v2
Hypothesis: Trade 4h Donchian(20) breakouts with 1d EMA50 trend filter and volume confirmation.
Uses ATR-based volume spike filter (volume > 2.0 * ATR) and discrete sizing (0.25).
Tightened entry conditions to reduce trades and minimize fee drag while maintaining edge in bull/bear markets.
Target: 20-30 trades/year to avoid overtrading and improve test generalization.
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
    
    # Get 1d data for trend filter and Donchian calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian(20) channels from 1d data
    donchian_high = pd.Series(df_1d['high'].values).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(df_1d['low'].values).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Calculate ATR for volume spike filter (adaptive to volatility)
    tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr2 = np.maximum(np.abs(low[1:] - close[:-1]), tr1)
    tr = np.concatenate([[np.inf], tr2])  # first TR undefined
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0  # track holding period
    
    # Start index: need warmup for 1d EMA50 (50) and Donchian (20) and ATR (14)
    start_idx = max(50, 20, 14, 1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            bars_since_entry = 0
            continue
        
        # Volume spike: current volume > 2.0 * ATR (adaptive threshold)
        volume_spike = volume[i] > 2.0 * atr[i]
        
        if position == 0:
            # Long: price breaks above Donchian high AND 1d trend bullish (close > EMA50) AND volume spike
            long_setup = (close[i] > donchian_high_aligned[i]) and \
                         (close[i] > ema_50_1d_aligned[i]) and \
                         volume_spike
            # Short: price breaks below Donchian low AND 1d trend bearish (close < EMA50) AND volume spike
            short_setup = (close[i] < donchian_low_aligned[i]) and \
                          (close[i] < ema_50_1d_aligned[i]) and \
                          volume_spike
            
            if long_setup:
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            elif short_setup:
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
            else:
                signals[i] = 0.0
                bars_since_entry = 0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            bars_since_entry += 1
            # Exit: price re-enters Donchian channel OR 1d trend turns bearish OR min holding period (6 bars = 1 day)
            if (donchian_low_aligned[i] <= close[i] <= donchian_high_aligned[i]) or \
               (close[i] < ema_50_1d_aligned[i]) or \
               (bars_since_entry >= 6):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            bars_since_entry += 1
            # Exit: price re-enters Donchian channel OR 1d trend turns bullish OR min holding period (6 bars = 1 day)
            if (donchian_low_aligned[i] <= close[i] <= donchian_high_aligned[i]) or \
               (close[i] > ema_50_1d_aligned[i]) or \
               (bars_since_entry >= 6):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
    
    return signals

name = "4h_Donchian20_Breakout_1dTrendFilter_VolumeSpike_v2"
timeframe = "4h"
leverage = 1.0