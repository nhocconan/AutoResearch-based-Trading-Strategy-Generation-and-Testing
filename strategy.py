#!/usr/bin/env python3
"""
12h_VORTEX_Breakout_1dTrend_Volume
Hypothesis: Vortex indicator identifies trend direction on 12h timeframe. Combining with 1d trend filter and volume spikes captures strong trends while avoiding whipsaws. Works in bull markets via long breakouts and bear markets via short breakdowns. Targets ~20-30 trades/year on 12h to minimize fee drag.
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Vortex indicator on 12h data
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = np.abs(high_12h[1:] - low_12h[:-1])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # Align with original indices
    
    # Vortex Indicator
    vm_plus = np.abs(high_12h[1:] - low_12h[:-1])
    vm_minus = np.abs(low_12h[1:] - high_12h[:-1])
    vm_plus = np.concatenate([[np.nan], vm_plus])
    vm_minus = np.concatenate([[np.nan], vm_minus])
    
    # Sum over 34 periods
    sum_tr = pd.Series(tr).rolling(window=34, min_periods=34).sum().values
    sum_vm_plus = pd.Series(vm_plus).rolling(window=34, min_periods=34).sum().values
    sum_vm_minus = pd.Series(vm_minus).rolling(window=34, min_periods=34).sum().values
    
    vi_plus = sum_vm_plus / sum_tr
    vi_minus = sum_vm_minus / sum_tr
    
    # Align Vortex to 12h timeframe
    vi_plus_aligned = align_htf_to_ltf(prices, df_12h, vi_plus)
    vi_minus_aligned = align_htf_to_ltf(prices, df_12h, vi_minus)
    
    # Volume confirmation: volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for Vortex and volume MA
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(vi_plus_aligned[i]) or np.isnan(vi_minus_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        vi_plus_val = vi_plus_aligned[i]
        vi_minus_val = vi_minus_aligned[i]
        ema_trend = ema34_1d_aligned[i]
        vol_spike_val = vol_spike[i]
        
        if position == 0:
            # Long: VI+ > VI- (uptrend) with 1d uptrend and volume spike
            if vi_plus_val > vi_minus_val and close[i] > ema_trend and vol_spike_val:
                signals[i] = size
                position = 1
            # Short: VI- > VI+ (downtrend) with 1d downtrend and volume spike
            elif vi_minus_val > vi_plus_val and close[i] < ema_trend and vol_spike_val:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: trend weakens or price breaks below 1d EMA
            if vi_plus_val <= vi_minus_val or close[i] < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: trend weakens or price breaks above 1d EMA
            if vi_minus_val <= vi_plus_val or close[i] > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_VORTEX_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0