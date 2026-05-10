#!/usr/bin/env python3
"""
1d_Vortex_EMA34_Volume
Hypothesis: The Vortex Indicator identifies trend direction, and when combined with EMA34 trend filter and volume spike on the daily timeframe, it captures strong trend moves with institutional interest. Designed for daily timeframe to limit trades to 10-25/year, reducing fee drain while performing in both bull and bear markets via trend alignment and volume confirmation.
"""

name = "1d_Vortex_EMA34_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data for Vortex, EMA34, and volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Vortex Indicator (14-period)
    # VM+ = |High - Prev Low|
    # VM- = |Low - Prev High|
    # TR = True Range
    # VI+ = Sum(VM+ over n) / Sum(TR over n)
    # VI- = Sum(VM- over n) / Sum(TR over n)
    vm_plus = np.abs(high_1d - np.roll(low_1d, 1))
    vm_minus = np.abs(low_1d - np.roll(high_1d, 1))
    tr1 = np.abs(high_1d - np.roll(low_1d, 1))
    tr2 = np.abs(low_1d - np.roll(high_1d, 1))
    tr3 = np.abs(np.roll(high_1d, 1) - np.roll(low_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Handle first element
    vm_plus[0] = np.abs(high_1d[0] - low_1d[0])
    vm_minus[0] = np.abs(low_1d[0] - high_1d[0])
    tr[0] = np.abs(high_1d[0] - low_1d[0])
    
    # Sum over 14 periods
    n_vortex = 14
    vm_plus_sum = pd.Series(vm_plus).rolling(window=n_vortex, min_periods=n_vortex).sum().values
    vm_minus_sum = pd.Series(vm_minus).rolling(window=n_vortex, min_periods=n_vortex).sum().values
    tr_sum = pd.Series(tr).rolling(window=n_vortex, min_periods=n_vortex).sum().values
    
    vi_plus = vm_plus_sum / tr_sum
    vi_minus = vm_minus_sum / tr_sum
    
    # EMA34 trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # 20-day average volume for spike detection
    vol_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to daily (no additional alignment needed as we're on 1d timeframe)
    vi_plus_aligned = vi_plus
    vi_minus_aligned = vi_minus
    ema34_1d_aligned = ema34_1d
    vol_avg_1d_aligned = vol_avg_1d
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Vortex (14), EMA34 (34), volume average (20)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(vi_plus_aligned[i]) or 
            np.isnan(vi_minus_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(vol_avg_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: EMA34
        uptrend_1d = close_1d[i] > ema34_1d_aligned[i]
        downtrend_1d = close_1d[i] < ema34_1d_aligned[i]
        
        # Volume filter: current day volume > 2.0x average volume
        volume_spike = volume_1d[i] > vol_avg_1d_aligned[i] * 2.0
        
        if position == 0:
            # Long entry: VI+ > VI- (bullish vortex) + uptrend + volume spike
            if vi_plus_aligned[i] > vi_minus_aligned[i] and uptrend_1d and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short entry: VI- > VI+ (bearish vortex) + downtrend + volume spike
            elif vi_minus_aligned[i] > vi_plus_aligned[i] and downtrend_1d and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: vortex turns bearish OR trend fails
            if vi_minus_aligned[i] > vi_plus_aligned[i] or not uptrend_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: vortex turns bullish OR trend fails
            if vi_plus_aligned[i] > vi_minus_aligned[i] or not downtrend_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals