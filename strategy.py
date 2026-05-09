#!/usr/bin/env python3
# 12h_Vortex_Trend_Confirmation_With_Volume_Spike
# Hypothesis: Use Vortex Indicator (VI) from 1d timeframe to identify trend direction on 12h chart.
# Long when VI+ > VI- and price breaks above 12h EMA20 with volume > 2x 20-period average.
# Short when VI- > VI+ and price breaks below 12h EMA20 with volume > 2x 20-period average.
# Exit when Vortex signal reverses or volume drops below average.
# Works in bull markets by catching strong uptrends, in bear markets by capturing downtrends.
# Volume spike ensures only high-momentum moves trigger entries, reducing false signals.
# Target: 15-30 trades/year on 12h timeframe to avoid fee drag.

name = "12h_Vortex_Trend_Confirmation_With_Volume_Spike"
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
    
    # Get 1d data for Vortex Indicator and EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Vortex Indicator components on 1d data
    # VM+ = |high(t) - low(t-1)|, VM- = |low(t) - high(t-1)|
    vm_plus = np.abs(high_1d - np.roll(low_1d, 1))
    vm_minus = np.abs(low_1d - np.roll(high_1d, 1))
    vm_plus[0] = np.nan  # First value has no previous
    vm_minus[0] = np.nan
    
    # True Range = max(|high-low|, |high-close_prev|, |low-close_prev|)
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    true_range = np.maximum(tr1, np.maximum(tr2, tr3))
    true_range[0] = np.nan  # First value has no previous close
    
    # Smooth over 14 periods (standard VI period)
    def smooth_sum(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        # Initialize first value
        result[period-1] = np.nansum(arr[0:period])
        # Rolling sum
        for i in range(period, len(arr)):
            if not np.isnan(result[i-1]) and not np.isnan(arr[i]):
                result[i] = result[i-1] - arr[i-period] + arr[i]
            else:
                result[i] = np.nan
        return result
    
    vm_plus_sum = smooth_sum(vm_plus, 14)
    vm_minus_sum = smooth_sum(vm_minus, 14)
    tr_sum = smooth_sum(true_range, 14)
    
    # VI+ = VM+_sum / TR_sum, VI- = VM-_sum / TR_sum
    vi_plus = np.full_like(high_1d, np.nan)
    vi_minus = np.full_like(high_1d, np.nan)
    valid = (~np.isnan(vm_plus_sum)) & (~np.isnan(vm_minus_sum)) & (~np.isnan(tr_sum)) & (tr_sum != 0)
    vi_plus[valid] = vm_plus_sum[valid] / tr_sum[valid]
    vi_minus[valid] = vm_minus_sum[valid] / tr_sum[valid]
    
    # Calculate 1d EMA20 for trend confirmation
    ema_20_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 20:
        ema_20_1d[19] = np.mean(close_1d[0:20])
        for i in range(20, len(close_1d)):
            ema_20_1d[i] = (close_1d[i] * 2 + ema_20_1d[i-1] * 18) / 20
    
    # Align 1d indicators to 12h timeframe
    vi_plus_aligned = align_htf_to_ltf(prices, df_1d, vi_plus)
    vi_minus_aligned = align_htf_to_ltf(prices, df_1d, vi_minus)
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # Calculate 12h EMA20 for entry trigger
    ema_20_12h = np.full_like(close, np.nan)
    if len(close) >= 20:
        ema_20_12h[19] = np.mean(close[0:20])
        for i in range(20, len(close)):
            ema_20_12h[i] = (close[i] * 2 + ema_20_12h[i-1] * 18) / 20
    
    # Volume filter: 12h volume / 20-period average volume
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid_vol = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid_vol] = volume[valid_vol] / vol_ma[valid_vol]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 1)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(vi_plus_aligned[i]) or np.isnan(vi_minus_aligned[i]) or \
           np.isnan(ema_20_1d_aligned[i]) or np.isnan(ema_20_12h[i]) or \
           np.isnan(volume_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: VI+ > VI- (bullish trend) AND price above EMA20 AND volume spike
            if vi_plus_aligned[i] > vi_minus_aligned[i] and close[i] > ema_20_12h[i] and volume_ratio[i] > 2.0:
                signals[i] = 0.25
                position = 1
            # Enter short: VI- > VI+ (bearish trend) AND price below EMA20 AND volume spike
            elif vi_minus_aligned[i] > vi_plus_aligned[i] and close[i] < ema_20_12h[i] and volume_ratio[i] > 2.0:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Vortex turns bearish OR price breaks below EMA20 OR volume drops
            if vi_minus_aligned[i] > vi_plus_aligned[i] or close[i] < ema_20_12h[i] or volume_ratio[i] < 1.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Vortex turns bullish OR price breaks above EMA20 OR volume drops
            if vi_plus_aligned[i] > vi_minus_aligned[i] or close[i] > ema_20_12h[i] or volume_ratio[i] < 1.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals