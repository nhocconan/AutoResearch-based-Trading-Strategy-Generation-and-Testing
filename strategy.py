# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
12h_Vortex_VolumeSpike_TrendFilter_v1
Hypothesis: Use Vortex indicator (VI+ and VI-) to identify trend direction on 12h, enter on Vortex crossover with volume confirmation (>2x 20-bar average). Exit when Vortex reverses or volume dries up. Trend filter: only trade in direction of daily EMA34 to avoid counter-trend trades. Designed for low trade frequency (12-37/year) to minimize fee drag. Works in both bull and bear markets by aligning with higher timeframe trend.
"""

name = "12h_Vortex_VolumeSpike_TrendFilter_v1"
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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate weekly data for additional trend confirmation (optional)
    df_1w = get_htf_data(prices, '1w')
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Vortex Indicator (VI) on 12h
    # VM+ = |high - low_prev|, VM- = |low - high_prev|
    # Sum over n periods (typically 14)
    n_vortex = 14
    vm_plus = np.abs(high - np.roll(low, 1))
    vm_minus = np.abs(low - np.roll(high, 1))
    # First value is invalid due to roll, set to 0
    vm_plus[0] = 0
    vm_minus[0] = 0
    
    # True Range for VI denominator
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first TR is just high-low
    
    # Sum over n_vortex periods
    vm_plus_sum = pd.Series(vm_plus).rolling(window=n_vortex, min_periods=n_vortex).sum().values
    vm_minus_sum = pd.Series(vm_minus).rolling(window=n_vortex, min_periods=n_vortex).sum().values
    tr_sum = pd.Series(tr).rolling(window=n_vortex, min_periods=n_vortex).sum().values
    
    # VI+ and VI-
    vi_plus = np.divide(vm_plus_sum, tr_sum, out=np.zeros_like(vm_plus_sum), where=tr_sum!=0)
    vi_minus = np.divide(vm_minus_sum, tr_sum, out=np.zeros_like(vm_minus_sum), where=tr_sum!=0)
    
    # Volume confirmation: 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.divide(volume, vol_ma20, out=np.zeros_like(volume), where=vol_ma20!=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, n_vortex)  # Warmup for volume MA and Vortex
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(vi_plus[i]) or np.isnan(vi_minus[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(ema_34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get daily close for trend determination
        daily_close_aligned = align_htf_to_ltf(prices, df_1d, df_1d['close'].values)
        if np.isnan(daily_close_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        daily_trend_up = daily_close_aligned[i] > ema_34_1d_aligned[i]
        daily_trend_down = daily_close_aligned[i] < ema_34_1d_aligned[i]
        
        # Weekly trend filter (optional, can be removed if too restrictive)
        weekly_trend_up = True
        weekly_trend_down = True
        if len(df_1w) > 0:
            weekly_close_aligned = align_htf_to_ltf(prices, df_1w, df_1w['close'].values)
            if not np.isnan(weekly_close_aligned[i]) and not np.isnan(ema_34_1w_aligned[i]):
                weekly_trend_up = weekly_close_aligned[i] > ema_34_1w_aligned[i]
                weekly_trend_down = weekly_close_aligned[i] < ema_34_1w_aligned[i]
        
        if position == 0:
            # Long: VI+ crosses above VI-, volume spike, daily trend up, weekly trend up
            if (vi_plus[i] > vi_minus[i] and vi_plus[i-1] <= vi_minus[i-1] and
                vol_ratio[i] > 2.0 and 
                daily_trend_up and 
                weekly_trend_up):
                signals[i] = 0.25
                position = 1
            # Short: VI- crosses above VI+, volume spike, daily trend down, weekly trend down
            elif (vi_minus[i] > vi_plus[i] and vi_minus[i-1] <= vi_plus[i-1] and
                  vol_ratio[i] > 2.0 and 
                  daily_trend_down and 
                  weekly_trend_down):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: VI- crosses above VI+ or trend changes
            if (vi_minus[i] > vi_plus[i] and vi_minus[i-1] <= vi_plus[i-1]) or not (daily_trend_up and weekly_trend_up):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: VI+ crosses above VI- or trend changes
            if (vi_plus[i] > vi_minus[i] and vi_plus[i-1] <= vi_minus[i-1]) or not (daily_trend_down and weekly_trend_down):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals