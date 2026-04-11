#!/usr/bin/env python3
# 4h_1d_vortex_volume_v1
# Strategy: 4h Vortex indicator breakout with 1d volume and ADX trend filter
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Vortex identifies trend initiation. When VI+ crosses above VI- with above-average
# 1d volume and ADX > 25, it signals strong trend initiation. Works in bull (VI+ > VI-) and
# bear (VI- > VI+) markets. Volume and trend filters reduce false signals. Target: 20-50 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_vortex_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Vortex indicator
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Vortex Indicator components
    vm_plus = np.abs(high_1d - np.concatenate([[low_1d[0]], low_1d[:-1]]))
    vm_minus = np.abs(low_1d - np.concatenate([[high_1d[0]], high_1d[:-1]]))
    
    # Smoothing periods (typically 14)
    period = 14
    
    def wilders_smooth(data, period):
        result = np.zeros_like(data)
        if len(data) < period:
            return result
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    tr14 = wilders_smooth(tr, period)
    vm_plus_14 = wilders_smooth(vm_plus, period)
    vm_minus_14 = wilders_smooth(vm_minus, period)
    
    # VI+ and VI-
    vi_plus = np.where(tr14 != 0, vm_plus_14 / tr14, 0)
    vi_minus = np.where(tr14 != 0, vm_minus_14 / tr14, 0)
    
    # Calculate ADX on 1d for trend strength
    # Directional Movement
    dm_plus = np.where((high_1d - np.concatenate([[high_1d[0]], high_1d[:-1]])) > 
                       (np.concatenate([[low_1d[0]], low_1d[:-1]]) - low_1d), 
                       np.maximum(high_1d - np.concatenate([[high_1d[0]], high_1d[:-1]]), 0), 0)
    dm_minus = np.where((np.concatenate([[low_1d[0]], low_1d[:-1]]) - low_1d) > 
                        (high_1d - np.concatenate([[high_1d[0]], high_1d[:-1]])), 
                        np.maximum(np.concatenate([[low_1d[0]], low_1d[:-1]]) - low_1d, 0), 0)
    
    dm_plus_14 = wilders_smooth(dm_plus, period)
    dm_minus_14 = wilders_smooth(dm_minus, period)
    
    # DI+ and DI-
    di_plus = np.where(tr14 != 0, 100 * dm_plus_14 / tr14, 0)
    di_minus = np.where(tr14 != 0, 100 * dm_minus_14 / tr14, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 
                  100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilders_smooth(dx, period)
    
    # Align indicators to 4h
    vi_plus_aligned = align_htf_to_ltf(prices, df_1d, vi_plus)
    vi_minus_aligned = align_htf_to_ltf(prices, df_1d, vi_minus)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # 1d volume average (20-period) for confirmation
    vol_1d = df_1d['volume'].values
    vol_avg_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if any required data is invalid
        if (np.isnan(vi_plus_aligned[i]) or np.isnan(vi_minus_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_avg_20_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Current 1d volume (aligned)
        vol_1d_current = align_htf_to_ltf(prices, df_1d, vol_1d)[i]
        vol_confirm = vol_1d_current > 1.5 * vol_avg_20_aligned[i]
        
        # Trend filter: ADX > 25 indicates strong trend
        trend_filter = adx_aligned[i] > 25
        
        # Vortex crossover conditions with volume and trend confirmation
        # Long: VI+ crosses above VI- (VI+ > VI- and previous VI+ <= previous VI-)
        # Short: VI- crosses above VI+ (VI- > VI+ and previous VI- <= previous VI+)
        if i > 0:
            vi_plus_cross_above = vi_plus_aligned[i] > vi_minus_aligned[i] and vi_plus_aligned[i-1] <= vi_minus_aligned[i-1]
            vi_minus_cross_above = vi_minus_aligned[i] > vi_plus_aligned[i] and vi_minus_aligned[i-1] <= vi_plus_aligned[i-1]
        else:
            vi_plus_cross_above = False
            vi_minus_cross_above = False
        
        long_signal = vi_plus_cross_above and vol_confirm and trend_filter
        short_signal = vi_minus_cross_above and vol_confirm and trend_filter
        
        # Exit conditions: reverse Vortex crossover OR trend weakening
        if i > 0:
            vi_minus_cross_above_vi_plus = vi_minus_aligned[i] > vi_plus_aligned[i] and vi_minus_aligned[i-1] <= vi_plus_aligned[i-1]
            vi_plus_cross_above_vi_minus = vi_plus_aligned[i] > vi_minus_aligned[i] and vi_plus_aligned[i-1] <= vi_minus_aligned[i-1]
        else:
            vi_minus_cross_above_vi_plus = False
            vi_plus_cross_above_vi_minus = False
        
        long_exit = vi_minus_cross_above_vi_plus or (adx_aligned[i] < 20)
        short_exit = vi_plus_cross_above_vi_minus or (adx_aligned[i] < 20)
        
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals