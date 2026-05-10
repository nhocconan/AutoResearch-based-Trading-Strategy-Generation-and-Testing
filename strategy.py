#!/usr/bin/env python3
# 12H_Vortex_1dTrend_VolumeSpike
# Hypothesis: Uses Vortex indicator on daily timeframe to capture trend direction and momentum,
# combined with volume spikes on 12h chart to confirm breakouts. Vortex (VI+ and VI-) helps identify
# the start of new trends and works in both bull and bear markets by filtering trades with the
# dominant daily trend. Volume spikes reduce false signals. Targets 12-37 trades per year on 12h.

name = "12H_Vortex_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Vortex trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Vortex Indicator (VI+ and VI-) on daily data
    # True Range
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Positive and Negative Vortex Movements
    vm_plus = np.abs(df_1d['high'] - df_1d['low'].shift(1))
    vm_minus = np.abs(df_1d['low'] - df_1d['high'].shift(1))
    
    # Sum over 14 periods (standard Vortex period)
    period = 14
    tr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    vm_plus_sum = pd.Series(vm_plus).rolling(window=period, min_periods=period).sum().values
    vm_minus_sum = pd.Series(vm_minus).rolling(window=period, min_periods=period).sum().values
    
    # VI+ and VI- (avoid division by zero)
    vi_plus = np.divide(vm_plus_sum, tr_sum, out=np.zeros_like(tr_sum), where=tr_sum!=0)
    vi_minus = np.divide(vm_minus_sum, tr_sum, out=np.zeros_like(tr_sum), where=tr_sum!=0)
    
    # Align Vortex to 12h timeframe (use prior day's values)
    vi_plus_aligned = align_htf_to_ltf(prices, df_1d, vi_plus)
    vi_minus_aligned = align_htf_to_ltf(prices, df_1d, vi_minus)
    
    # Volume filter: volume > 2.0x 20-period average on 12h chart
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma * 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Warmup for Vortex and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(vi_plus_aligned[i]) or np.isnan(vi_minus_aligned[i]) or np.isnan(vol_threshold[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: VI+ > VI- indicates uptrend, VI- > VI+ indicates downtrend
        vi_plus_gt = vi_plus_aligned[i] > vi_minus_aligned[i]
        vi_minus_gt = vi_minus_aligned[i] > vi_plus_aligned[i]
        
        if position == 0:
            # Long entry: VI+ > VI- (uptrend) + volume spike
            if vi_plus_gt and volume[i] > vol_threshold[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: VI- > VI+ (downtrend) + volume spike
            elif vi_minus_gt and volume[i] > vol_threshold[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend reverses (VI- > VI+) or volume drops below average
            if vi_minus_gt or volume[i] < vol_ma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend reverses (VI+ > VI-) or volume drops below average
            if vi_plus_gt or volume[i] < vol_ma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals