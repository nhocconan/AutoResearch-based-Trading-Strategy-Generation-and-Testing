#!/usr/bin/env python3
# 4h_Vortex_Trend_Filter_Signal
# Strategy: Uses Vortex Indicator to detect trend direction, filtered by 1d EMA(50) and volume confirmation.
# Long when VI+ > VI- and price > 1d EMA50 and volume > 1.5x 20-period average.
# Short when VI- > VI+ and price < 1d EMA50 and volume > 1.5x 20-period average.
# Exit when Vortex crossover reverses or volume condition fails.
# Designed for 4h timeframe with low trade frequency to avoid fee drag, works in both bull and bear markets via trend-following logic.

name = "4h_Vortex_Trend_Filter_Signal"
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
    
    # Get 1d EMA(50) for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate Vortex Indicator (VI) over 14 periods
    # True Range components
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with original index
    
    vm_plus = np.abs(high - np.roll(low, 1))
    vm_minus = np.abs(low - np.roll(high, 1))
    vm_plus = np.concatenate([[np.nan], vm_plus[1:]])
    vm_minus = np.concatenate([[np.nan], vm_minus[1:]])
    
    # Sum over 14 periods
    n_period = 14
    tr_sum = np.full_like(tr, np.nan)
    vm_plus_sum = np.full_like(vm_plus, np.nan)
    vm_minus_sum = np.full_like(vm_minus, np.nan)
    
    for i in range(n_period, len(tr)):
        tr_sum[i] = np.nansum(tr[i-n_period+1:i+1])
        vm_plus_sum[i] = np.nansum(vm_plus[i-n_period+1:i+1])
        vm_minus_sum[i] = np.nansum(vm_minus[i-n_period+1:i+1])
    
    vi_plus = np.where(tr_sum != 0, vm_plus_sum / tr_sum, 0)
    vi_minus = np.where(tr_sum != 0, vm_minus_sum / tr_sum, 0)
    
    # Volume filter: current volume > 1.5 x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    for i in range(20, len(volume)):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, n_period)  # ensure indicators are ready
    
    for i in range(start_idx, n):
        if np.isnan(vi_plus[i]) or np.isnan(vi_minus[i]) or np.isnan(ema_50_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: VI+ > VI-, price above EMA50, volume confirmation
            if vi_plus[i] > vi_minus[i] and close[i] > ema_50_aligned[i] and vol_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: VI- > VI+, price below EMA50, volume confirmation
            elif vi_minus[i] > vi_plus[i] and close[i] < ema_50_aligned[i] and vol_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: VI- crosses above VI+ or volume filter fails
            if vi_minus[i] >= vi_plus[i] or not vol_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: VI+ crosses above VI- or volume filter fails
            if vi_plus[i] >= vi_minus[i] or not vol_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals