#!/usr/bin/env python3
# 12h_Vortex_1dTrend
# Hypothesis: Use Vortex indicator (VI+ and VI-) on 12h to detect trend direction,
# filtered by 1d EMA34 trend and volume spike. Enter long when VI+ > VI- and price > 1d EMA34,
# enter short when VI- > VI+ and price < 1d EMA34. Exit on Vortex crossover or trend failure.
# Designed for low frequency (10-30 trades/year) to avoid fee drag. Works in bull (catch trends)
# and bear (catch downtrends) with trend filter and volume confirmation.

name = "12h_Vortex_1dTrend"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def vortex_indicator(high, low, close):
    """
    Calculate Vortex Indicator (VI+ and VI-).
    Returns VI+ and VI- arrays.
    """
    n = len(high)
    tr = np.zeros(n)
    vm_plus = np.zeros(n)
    vm_minus = np.zeros(n)
    
    # True Range
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Vortex Movement
    for i in range(1, n):
        vm_plus[i] = abs(high[i] - low[i-1])
        vm_minus[i] = abs(low[i] - high[i-1])
    
    # Sum over period (default 14)
    period = 14
    tr_sum = np.zeros(n)
    vm_plus_sum = np.zeros(n)
    vm_minus_sum = np.zeros(n)
    
    for i in range(n):
        if i < period:
            tr_sum[i] = np.sum(tr[max(0, i-period+1):i+1])
            vm_plus_sum[i] = np.sum(vm_plus[max(0, i-period+1):i+1])
            vm_minus_sum[i] = np.sum(vm_minus[max(0, i-period+1):i+1])
        else:
            tr_sum[i] = tr_sum[i-1] - tr[i-period] + tr[i]
            vm_plus_sum[i] = vm_plus_sum[i-1] - vm_plus[i-period] + vm_plus[i]
            vm_minus_sum[i] = vm_minus_sum[i-1] - vm_minus[i-period] + vm_minus[i]
    
    # Avoid division by zero
    vi_plus = np.where(tr_sum > 0, vm_plus_sum / tr_sum, 0)
    vi_minus = np.where(tr_sum > 0, vm_minus_sum / tr_sum, 0)
    
    return vi_plus, vi_minus

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Vortex on 12h data
    vi_plus, vi_minus = vortex_indicator(high, low, close)
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume confirmation: 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align daily data to 12h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(vi_plus[i]) or np.isnan(vi_minus[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter: price above/below daily EMA34
        trend_up = close[i] > ema_34_1d_aligned[i]
        trend_down = close[i] < ema_34_1d_aligned[i]
        
        # Volume filter
        vol_ok = volume[i] > vol_ma_20[i]
        
        # Vortex signals
        vi_plus_cross = vi_plus[i] > vi_minus[i]
        vi_minus_cross = vi_minus[i] > vi_plus[i]
        
        if position == 0:
            # LONG: VI+ > VI-, price above daily EMA34, volume confirmation
            if vi_plus_cross and trend_up and vol_ok:
                signals[i] = 0.25
                position = 1
            # SHORT: VI- > VI+, price below daily EMA34, volume confirmation
            elif vi_minus_cross and trend_down and vol_ok:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: VI- > VI+ (Vortex crossover down) or trend fails
            if vi_minus_cross or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: VI+ > VI- (Vortex crossover up) or trend fails
            if vi_plus_cross or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals