#!/usr/bin/env python3
"""
12h_Vortex_Trend_Filter_1dVWAP_Support_Resistance
Hypothesis: Use Vortex Indicator (VI) for trend direction on 12h timeframe, filtered by daily VWAP
as dynamic support/resistance. Enter long when VI+ > VI- and price > daily VWAP, short when VI- > VI+
and price < daily VWAP. Exit on opposite signal. Designed for low trade frequency with strong trend
filtering to work in both bull and bear markets.
"""

name = "12h_Vortex_Trend_Filter_1dVWAP_Support_Resistance"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for VWAP calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily VWAP (typical price * volume cumulative / volume cumulative)
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    pv_1d = typical_price_1d * volume_1d
    cum_pv_1d = np.cumsum(pv_1d)
    cum_volume_1d = np.cumsum(volume_1d)
    vwap_1d = np.divide(cum_pv_1d, cum_volume_1d, out=np.full_like(cum_pv_1d, np.nan), where=cum_volume_1d!=0)
    
    # Align daily VWAP to 12h timeframe
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # Calculate Vortex Indicator on 12h data (period=14)
    vm_plus = np.abs(high - np.roll(low, 1))
    vm_minus = np.abs(low - np.roll(high, 1))
    vm_plus[0] = np.abs(high[0] - low[0])  # First period
    vm_minus[0] = np.abs(low[0] - high[0])  # First period
    
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]  # First period
    
    # Sum over 14 periods
    n_periods = 14
    vm_plus_sum = np.full_like(vm_plus, np.nan)
    vm_minus_sum = np.full_like(vm_minus, np.nan)
    tr_sum = np.full_like(tr, np.nan)
    
    if len(vm_plus) >= n_periods:
        vm_plus_sum[n_periods-1] = np.sum(vm_plus[0:n_periods])
        vm_minus_sum[n_periods-1] = np.sum(vm_minus[0:n_periods])
        tr_sum[n_periods-1] = np.sum(tr[0:n_periods])
        for i in range(n_periods, len(vm_plus)):
            vm_plus_sum[i] = vm_plus_sum[i-1] - vm_plus[i-n_periods] + vm_plus[i]
            vm_minus_sum[i] = vm_minus_sum[i-1] - vm_minus[i-n_periods] + vm_minus[i]
            tr_sum[i] = tr_sum[i-1] - tr[i-n_periods] + tr[i]
    
    vi_plus = np.divide(vm_plus_sum, tr_sum, out=np.full_like(vm_plus_sum, np.nan), where=tr_sum!=0)
    vi_minus = np.divide(vm_minus_sum, tr_sum, out=np.full_like(vm_minus_sum, np.nan), where=tr_sum!=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(n_periods-1, 0)  # Ensure VI is ready
    
    for i in range(start_idx, n):
        # Skip if VWAP or VI data not ready
        if (np.isnan(vwap_1d_aligned[i]) or np.isnan(vi_plus[i]) or np.isnan(vi_minus[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: VI+ > VI- (bullish trend) AND price > VWAP (above support)
            if vi_plus[i] > vi_minus[i] and close[i] > vwap_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: VI- > VI+ (bearish trend) AND price < VWAP (below resistance)
            elif vi_minus[i] > vi_plus[i] and close[i] < vwap_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: VI- > VI+ (trend turns bearish) OR price < VWAP (breaks support)
            if vi_minus[i] > vi_plus[i] or close[i] < vwap_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: VI+ > VI- (trend turns bullish) OR price > VWAP (breaks resistance)
            if vi_plus[i] > vi_minus[i] or close[i] > vwap_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals