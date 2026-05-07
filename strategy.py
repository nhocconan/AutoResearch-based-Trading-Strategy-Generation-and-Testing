#!/usr/bin/env python3
"""
4h_Vortex_VolumeSpike_TrendFilter_v1
Hypothesis: 4h Vortex indicator (VI+ > VI-) for trend direction, combined with volume spike (>2x average) and price above/below 4h VWAP for entry confirmation. Uses 1d VWAP as higher timeframe trend filter. Designed for low trade frequency (20-50/year) to avoid fee drag. Works in both bull and bear markets by aligning with higher timeframe VWAP trend.
"""

name = "4h_Vortex_VolumeSpike_TrendFilter_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for VWAP trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily VWAP
    df_1d['vwap'] = (df_1d['close'] * df_1d['volume']).cumsum() / df_1d['volume'].cumsum()
    vwap_1d = df_1d['vwap'].values
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # Calculate Vortex Indicator (14 periods)
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # Align with original arrays
    
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
    vi_plus = np.divide(vm_plus_sum, tr_sum, out=np.zeros_like(tr_sum), where=tr_sum!=0)
    vi_minus = np.divide(vm_minus_sum, tr_sum, out=np.zeros_like(tr_sum), where=tr_sum!=0)
    
    # VWAP (4h)
    vwap = (close * volume).cumsum() / volume.cumsum()
    
    # Volume confirmation: 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.divide(volume, vol_ma20, out=np.zeros_like(volume), where=vol_ma20!=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Warmup for Vortex and volume
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(vi_plus[i]) or np.isnan(vi_minus[i]) or 
            np.isnan(vwap[i]) or np.isnan(vwap_1d_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine higher timeframe trend using daily VWAP
        daily_close = close[i]  # Current 4h close for trend comparison
        htf_trend_up = daily_close > vwap_1d_aligned[i]
        htf_trend_down = daily_close < vwap_1d_aligned[i]
        
        if position == 0:
            # Long: VI+ > VI-, price above VWAP, volume spike, HTF trend up
            if (vi_plus[i] > vi_minus[i] and 
                close[i] > vwap[i] and 
                vol_ratio[i] > 2.0 and 
                htf_trend_up):
                signals[i] = 0.25
                position = 1
            # Short: VI- > VI+, price below VWAP, volume spike, HTF trend down
            elif (vi_minus[i] > vi_plus[i] and 
                  close[i] < vwap[i] and 
                  vol_ratio[i] > 2.0 and 
                  htf_trend_down):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: VI- > VI+ or price below VWAP
            if vi_minus[i] > vi_plus[i] or close[i] < vwap[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: VI+ > VI- or price above VWAP
            if vi_plus[i] > vi_minus[i] or close[i] > vwap[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals