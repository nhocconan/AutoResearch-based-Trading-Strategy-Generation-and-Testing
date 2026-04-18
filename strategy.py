#!/usr/bin/env python3
"""
4h_Vortex_Trend_12hTrendFilter_V1
Hypothesis: Trend-following strategy using Vortex Indicator on 4h for entry/exit, filtered by 12h EMA34 trend direction. The Vortex Indicator identifies trend direction and strength by comparing positive and negative vortex movement. Combined with 12h EMA34 trend filter to avoid counter-trend trades, this should work in both bull and bear markets by following the higher timeframe trend. Volume confirmation (>1.5x 24-period average) adds robustness. Target: 20-40 trades/year via trend-following with higher timeframe filter.
"""

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
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # 12h EMA34 for trend filter
    ema_period = 34
    close_12h = df_12h['close'].values
    ema_12h = np.full_like(close_12h, np.nan)
    
    if len(close_12h) >= ema_period:
        # Calculate EMA using Wilder's smoothing (alpha = 1/period)
        alpha = 1.0 / ema_period
        ema_12h[0] = close_12h[0]
        for i in range(1, len(close_12h)):
            ema_12h[i] = alpha * close_12h[i] + (1 - alpha) * ema_12h[i-1]
    
    # Align 12h EMA to 4h timeframe
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Vortex Indicator on 4h
    vortex_period = 14
    
    # Calculate True Range
    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
    )
    # Prepend first TR value (high-low for first bar)
    tr = np.concatenate([[high[0] - low[0]], tr])
    
    # Calculate +VM and -VM
    vm_plus = np.abs(high[1:] - low[:-1])
    vm_minus = np.abs(low[1:] - high[:-1])
    # Prepend first values
    vm_plus = np.concatenate([[0], vm_plus])
    vm_minus = np.concatenate([[0], vm_minus])
    
    # Sum over vortex_period
    tr_sum = np.full_like(tr, np.nan)
    vm_plus_sum = np.full_like(vm_plus, np.nan)
    vm_minus_sum = np.full_like(vm_minus, np.nan)
    
    if len(tr) >= vortex_period:
        for i in range(vortex_period - 1, len(tr)):
            tr_sum[i] = np.sum(tr[i - vortex_period + 1:i + 1])
            vm_plus_sum[i] = np.sum(vm_plus[i - vortex_period + 1:i + 1])
            vm_minus_sum[i] = np.sum(vm_minus[i - vortex_period + 1:i + 1])
    
    # Calculate VI+ and VI-
    vi_plus = np.full_like(tr, np.nan)
    vi_minus = np.full_like(tr, np.nan)
    
    if len(tr) >= vortex_period:
        for i in range(vortex_period - 1, len(tr)):
            if tr_sum[i] != 0:
                vi_plus[i] = vm_plus_sum[i] / tr_sum[i]
                vi_minus[i] = vm_minus_sum[i] / tr_sum[i]
    
    # Align Vortex to 4h timeframe (same as input, but using alignment for consistency)
    df_4h = get_htf_data(prices, '4h')
    vi_plus_aligned = align_htf_to_ltf(prices, df_4h, vi_plus)
    vi_minus_aligned = align_htf_to_ltf(prices, df_4h, vi_minus)
    
    # Volume confirmation: volume > 1.5x 24-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 24
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(vortex_period, vol_period, 30)  # Ensure enough data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(vi_plus_aligned[i]) or np.isnan(vi_minus_aligned[i]) or 
            np.isnan(ema_12h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: VI+ > VI- (bullish trend) + 12h EMA above price (uptrend) + volume
            if vi_plus_aligned[i] > vi_minus_aligned[i] and close[i] > ema_12h_aligned[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: VI- > VI+ (bearish trend) + 12h EMA below price (downtrend) + volume
            elif vi_minus_aligned[i] > vi_plus_aligned[i] and close[i] < ema_12h_aligned[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: VI- > VI+ (trend reversal) or 12h EMA below price (trend change)
            if vi_minus_aligned[i] > vi_plus_aligned[i] or close[i] < ema_12h_aligned[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: VI+ > VI- (trend reversal) or 12h EMA above price (trend change)
            if vi_plus_aligned[i] > vi_minus_aligned[i] or close[i] > ema_12h_aligned[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Vortex_Trend_12hTrendFilter_V1"
timeframe = "4h"
leverage = 1.0