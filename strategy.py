#!/usr/bin/env python3
"""
4h_Vortex_Trend_1dVolumeSpike
Hypothesis: Combine Vortex Indicator (trend strength) with daily volume spikes for breakout confirmation.
In bull markets: VI+ > VI- signals uptrend, long on breakout above prior day high with volume spike.
In bear markets: VI- > VI+ signals downtrend, short on breakdown below prior day low with volume spike.
Uses daily volume filter to ensure institutional participation. Trend filter avoids whipsaws.
Designed for 20-30 trades/year, works in both regimes via trend confirmation.
"""

name = "4h_Vortex_Trend_1dVolumeSpike"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data for Vortex calculation and volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Vortex Indicator (VM+, VM-, VI+, VI-)
    high = df_1d['high'].values
    low = df_1d['low'].values
    close = df_1d['close'].values
    
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # Align with index
    
    # VM+ and VM-
    vm_plus = np.abs(high[1:] - low[:-1])
    vm_minus = np.abs(low[1:] - high[:-1])
    vm_plus = np.concatenate([[np.nan], vm_plus])
    vm_minus = np.concatenate([[np.nan], vm_minus])
    
    # Sum over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    vm_plus_sum = pd.Series(vm_plus).rolling(window=14, min_periods=14).sum().values
    vm_minus_sum = pd.Series(vm_minus).rolling(window=14, min_periods=14).sum().values
    
    # VI+ and VI-
    vi_plus = vm_plus_sum / tr_sum
    vi_minus = vm_minus_sum / tr_sum
    
    # Daily volume filter: current volume > 2.0x 20-day average
    volume = df_1d['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > vol_ma20 * 2.0
    
    # Prior day high/low for breakout levels
    prior_day_high = df_1d['high'].shift(1).values
    prior_day_low = df_1d['low'].shift(1).values
    
    # Align all to 4h timeframe
    vi_plus_aligned = align_htf_to_ltf(prices, df_1d, vi_plus)
    vi_minus_aligned = align_htf_to_ltf(prices, df_1d, vi_minus)
    prior_day_high_aligned = align_htf_to_ltf(prices, df_1d, prior_day_high)
    prior_day_low_aligned = align_htf_to_ltf(prices, df_1d, prior_day_low)
    volume_filter_aligned = align_htf_to_ltf(prices, df_1d, volume_filter)
    
    # Get 4h price data
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 14-period Vortex + 1-day shift + 20-day volume MA
    start_idx = 35
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(vi_plus_aligned[i]) or 
            np.isnan(vi_minus_aligned[i]) or
            np.isnan(prior_day_high_aligned[i]) or
            np.isnan(prior_day_low_aligned[i]) or
            np.isnan(volume_filter_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: VI+ > VI- (uptrend) AND break above prior day high with volume spike
            if vi_plus_aligned[i] > vi_minus_aligned[i] and high_4h[i] > prior_day_high_aligned[i] and volume_filter_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: VI- > VI+ (downtrend) AND break below prior day low with volume spike
            elif vi_minus_aligned[i] > vi_plus_aligned[i] and low_4h[i] < prior_day_low_aligned[i] and volume_filter_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend turns bearish OR price breaks below prior day low
            if vi_minus_aligned[i] > vi_plus_aligned[i] or low_4h[i] < prior_day_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend turns bullish OR price breaks above prior day high
            if vi_plus_aligned[i] > vi_minus_aligned[i] or high_4h[i] > prior_day_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals