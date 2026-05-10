#!/usr/bin/env python3
"""
4H_Vortex_1dTrend_Confirmation
Hypothesis: Uses Vortex indicator on 1d timeframe to detect strong trend direction, 
enters on 4h pullbacks to VWAP in direction of daily trend with volume confirmation.
Designed for low trade frequency (target: 20-40/year) to avoid fee drag.
Works in both bull and bear markets by following established daily trends.
"""

name = "4H_Vortex_1dTrend_Confirmation"
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
    
    # Daily data for Vortex trend direction
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Vortex Indicator (VI) on daily timeframe
    # VI+ and VI- to determine trend direction
    tr1 = np.maximum(high_1d[1:], close_1d[:-1]) - np.minimum(low_1d[1:], close_1d[:-1])
    tr1 = np.insert(tr1, 0, high_1d[0] - low_1d[0])  # First TR
    
    vm_plus = np.abs(high_1d[1:] - low_1d[:-1])
    vm_minus = np.abs(low_1d[1:] - high_1d[:-1])
    vm_plus = np.insert(vm_plus, 0, np.abs(high_1d[0] - low_1d[0]))
    vm_minus = np.insert(vm_minus, 0, np.abs(high_1d[0] - low_1d[0]))
    
    # Sum over 14 periods (standard Vortex period)
    period = 14
    def sum_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p-1, len(arr)):
                res[i] = np.sum(arr[i-p+1:i+1])
        return res
    
    tr14 = sum_arr(tr1, period)
    vm_plus_14 = sum_arr(vm_plus, period)
    vm_minus_14 = sum_arr(vm_minus, period)
    
    # Avoid division by zero
    vi_plus = np.where(tr14 != 0, vm_plus_14 / tr14, 0)
    vi_minus = np.where(tr14 != 0, vm_minus_14 / tr14, 0)
    
    # Trend direction: VI+ > VI- = uptrend, VI- > VI+ = downtrend
    vi_plus_minus_diff = vi_plus - vi_minus
    
    # Daily VWAP for 4h entry timing
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    vwap_1d = np.full_like(close_1d, np.nan)
    cum_vol = np.zeros_like(volume)
    cum_price_vol = np.zeros_like(volume)
    
    for i in range(len(close_1d)):
        cum_vol[i] = (cum_vol[i-1] if i > 0 else 0) + volume[i] if hasattr(volume, '__len__') else volume[i]  # This needs fixing
    
    # Simpler approach: use typical price * volume for VWAP calculation
    tp_vol = typical_price_1d * df_1d['volume'].values
    cum_tp_vol = np.nancumsum(tp_vol)
    cum_vol = np.nancumsum(df_1d['volume'].values)
    vwap_1d = np.where(cum_vol != 0, cum_tp_vol / cum_vol, typical_price_1d)
    
    # 4h VWAP for entry signals
    tp_4h = (high + low + close) / 3.0
    tp_vol_4h = tp_4h * volume
    cum_tp_vol_4h = np.nancumsum(tp_vol_4h)
    cum_vol_4h = np.nancumsum(volume)
    vwap_4h = np.where(cum_vol_4h != 0, cum_tp_vol_4h / cum_vol_4h, tp_4h)
    
    # Align daily indicators to 4h timeframe
    vi_plus_minus_aligned = align_htf_to_ltf(prices, df_1d, vi_plus_minus_diff)
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, period)  # Ensure sufficient data
    
    for i in range(start_idx, n):
        if np.isnan(vi_plus_minus_aligned[i]) or np.isnan(vwap_1d_aligned[i]) or np.isnan(vwap_4h[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend strength and direction from Vortex
        trend_strength = abs(vi_plus_minus_aligned[i])
        is_uptrend = vi_plus_minus_aligned[i] > 0.1  # Threshold to avoid choppy signals
        is_downtrend = vi_plus_minus_aligned[i] < -0.1
        
        # Price relative to VWAP for entry timing
        price_above_vwap = close[i] > vwap_4h[i] * 1.002  # Small buffer to avoid whipsaw
        price_below_vwap = close[i] < vwap_4h[i] * 0.998
        
        # Volume confirmation: current volume > 1.3x average volume
        vol_ma = np.mean(volume[max(0, i-24):i+1]) if i >= 24 else np.mean(volume[:i+1])
        volume_confirm = volume[i] > 1.3 * vol_ma if not np.isnan(vol_ma) else False
        
        if position == 0:
            # Long: daily uptrend, price pulls back to/below VWAP, then closes above with volume
            if is_uptrend and price_below_vwap and close[i] > vwap_4h[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: daily downtrend, price bounces up to/above VWAP, then closes below with volume
            elif is_downtrend and price_above_vwap and close[i] < vwap_4h[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend weakens or price breaks below VWAP significantly
            if not is_uptrend or close[i] < vwap_4h[i] * 0.995:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend weakens or price breaks above VWAP significantly
            if not is_downtrend or close[i] > vwap_4h[i] * 1.005:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals