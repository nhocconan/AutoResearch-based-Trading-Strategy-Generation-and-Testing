#!/usr/bin/env python3
"""
4h_Vortex_Trend_Strength_1dFilter
Hypothesis: Vortex Indicator identifies trend strength (VI+ > VI- for uptrend, VI- > VI+ for downtrend). 
Combined with 1d EMA50 trend filter and volume confirmation to avoid whipsaws. 
Works in bull markets (strong VI+) and bear markets (strong VI-). 
Target: 20-50 total trades over 4 years (5-12/year) to minimize fee drag.
"""

name = "4h_Vortex_Trend_Strength_1dFilter"
timeframe = "4h"
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
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema50_1d[49] = np.mean(close_1d[:50])
        alpha = 2 / (50 + 1)
        for i in range(50, len(close_1d)):
            ema50_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema50_1d[i-1]
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 1d volume SMA20 for volume confirmation
    volume_1d = df_1d['volume'].values
    vol_sma20_1d = np.full(len(volume_1d), np.nan)
    if len(volume_1d) >= 20:
        vol_sma20_1d[19] = np.mean(volume_1d[:20])
        for i in range(20, len(volume_1d)):
            vol_sma20_1d[i] = (vol_sma20_1d[i-1] * 19 + volume_1d[i]) / 20
    vol_sma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma20_1d)
    
    # Vortex Indicator (period=14) on 4h data
    vm_plus = np.abs(high - np.roll(low, 1))
    vm_minus = np.abs(low - np.roll(high, 1))
    vm_plus[0] = np.abs(high[0] - low[0])  # first period
    vm_minus[0] = np.abs(high[0] - low[0])
    
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]  # first period
    
    # Sum over last 14 periods
    n_v = 14
    vi_plus = np.zeros(n)
    vi_minus = np.zeros(n)
    
    for i in range(n_v, n):
        if np.isnan(tr[i-n_v:i]).any() or np.isnan(vm_plus[i-n_v:i]).any() or np.isnan(vm_minus[i-n_v:i]).any():
            vi_plus[i] = np.nan
            vi_minus[i] = np.nan
        else:
            sum_vm_plus = np.sum(vm_plus[i-n_v+1:i+1])
            sum_vm_minus = np.sum(vm_minus[i-n_v+1:i+1])
            sum_tr = np.sum(tr[i-n_v+1:i+1])
            if sum_tr > 0:
                vi_plus[i] = sum_vm_plus / sum_tr
                vi_minus[i] = sum_vm_minus / sum_tr
            else:
                vi_plus[i] = np.nan
                vi_minus[i] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, n_v)  # warmup
    
    for i in range(start_idx, n):
        if np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_sma20_1d_aligned[i]) or np.isnan(vi_plus[i]) or np.isnan(vi_minus[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x average 1d volume (scaled to 4h)
        vol_4h_approx = vol_sma20_1d_aligned[i] / 6.0  # 24h/4h = 6
        volume_confirm = volume[i] > 1.5 * vol_4h_approx
        
        if position == 0:
            # Long: VI+ > VI- (uptrend) + above 1d EMA50 + volume confirmation
            if vi_plus[i] > vi_minus[i] and close[i] > ema50_1d_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: VI- > VI+ (downtrend) + below 1d EMA50 + volume confirmation
            elif vi_minus[i] > vi_plus[i] and close[i] < ema50_1d_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Trend weakness (VI- > VI+) or trend reversal (below EMA50)
            if vi_minus[i] > vi_plus[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Trend weakness (VI+ > VI-) or trend reversal (above EMA50)
            if vi_plus[i] > vi_minus[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals