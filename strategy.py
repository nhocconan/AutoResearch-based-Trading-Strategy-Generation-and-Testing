#!/usr/bin/env python3
# 4h_Vortex_Trend_Strategy
# Hypothesis: Vortex indicator on 4h chart identifies strong trends (VI+ > VI- for uptrend, VI- > VI+ for downtrend).
# Filtered by 1d EMA200 to avoid counter-trend trades in strong trends.
# Uses 1d volume spike (volume > 1.5x 20-period average) to confirm institutional interest.
# Stops when Vortex signal weakens or reverses.
# Target: 15-40 trades/year (60-160 total over 4 years) to minimize fee drag.

name = "4h_Vortex_Trend_Strategy"
timeframe = "4h"
leverage = 1.0

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
    
    # Get 1d data for trend filter and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Calculate 1d volume spike confirmation
    vol_1d = df_1d['volume'].values
    vol_ma20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = vol_1d > (vol_ma20_1d * 1.5)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d.astype(float))
    
    # Calculate Vortex Indicator on 4h data
    period = 14
    
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Vortex Movement
    vm_plus = np.abs(high[1:] - low[:-1])  # |current high - previous low|
    vm_minus = np.abs(low[1:] - high[:-1])  # |current low - previous high|
    vm_plus = np.concatenate([[0], vm_plus])
    vm_minus = np.concatenate([[0], vm_minus])
    
    # Sum of values over period
    tr_sum = np.full_like(high, np.nan)
    vm_plus_sum = np.full_like(high, np.nan)
    vm_minus_sum = np.full_like(high, np.nan)
    
    for i in range(len(high)):
        if i >= period:
            tr_sum[i] = np.nansum(tr[i-period+1:i+1])
            vm_plus_sum[i] = np.nansum(vm_plus[i-period+1:i+1])
            vm_minus_sum[i] = np.nansum(vm_minus[i-period+1:i+1])
    
    # Vortex Indicator
    vi_plus = np.full_like(high, np.nan)
    vi_minus = np.full_like(high, np.nan)
    
    valid = ~np.isnan(tr_sum) & (tr_sum != 0)
    vi_plus[valid] = vm_plus_sum[valid] / tr_sum[valid]
    vi_minus[valid] = vm_minus_sum[valid] / tr_sum[valid]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = period
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(vi_plus[i]) or np.isnan(vi_minus[i]) or 
            np.isnan(ema200_1d_aligned[i]) or np.isnan(vol_spike_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Determine trend from 1d EMA200
            uptrend = close[i] > ema200_1d_aligned[i]
            downtrend = close[i] < ema200_1d_aligned[i]
            
            # Long: uptrend + VI+ > VI- + volume spike
            if uptrend and vi_plus[i] > vi_minus[i] and vol_spike_1d_aligned[i] > 0.5:
                signals[i] = 0.25
                position = 1
            # Short: downtrend + VI- > VI+ + volume spike
            elif downtrend and vi_minus[i] > vi_plus[i] and vol_spike_1d_aligned[i] > 0.5:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if trend weakens or reverses
            if vi_plus[i] <= vi_minus[i] or close[i] < ema200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if trend weakens or reverses
            if vi_minus[i] <= vi_plus[i] or close[i] > ema200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals