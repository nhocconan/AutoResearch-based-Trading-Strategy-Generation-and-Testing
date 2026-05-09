#!/usr/bin/env python3
# 4h_Vortex_Trend_Strength_12hTrend_VolumeSpike
# Hypothesis: Vortex indicator trend strength (VI+ > VI-) combined with 12h EMA50 trend filter and volume spike.
# Works in bull/bear: Trend filter avoids counter-trend trades, volume confirms momentum.
# Vortex captures trend initiation and continuation; filters reduce false signals and overtrading.

name = "4h_Vortex_Trend_Strength_12hTrend_VolumeSpike"
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
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = np.full_like(close_12h, np.nan)
    if len(close_12h) >= 50:
        ema_50_12h[49] = np.mean(close_12h[0:50])
        for i in range(50, len(close_12h)):
            ema_50_12h[i] = (ema_50_12h[i-1] * 49 + close_12h[i]) / 50
    
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Vortex Indicator (VI+ and VI-) - 14 periods
    tr = np.maximum(np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1])), np.abs(low[1:] - close[:-1]))
    tr = np.concatenate([[np.nan], tr])  # align with indices
    
    vm_plus = np.abs(high - low[:-1])
    vm_minus = np.abs(low - high[:-1])
    
    # Sum over 14 periods
    n_tr = len(tr)
    vi_plus = np.full(n_tr, np.nan)
    vi_minus = np.full(n_tr, np.nan)
    
    if n_tr >= 14:
        # Initial sums
        tr_sum = np.nansum(tr[1:15])  # skip first NaN
        vm_plus_sum = np.nansum(vm_plus[1:15])
        vm_minus_sum = np.nansum(vm_minus[1:15])
        
        if tr_sum > 0:
            vi_plus[14] = vm_plus_sum / tr_sum
            vi_minus[14] = vm_minus_sum / tr_sum
        
        # Rolling update
        for i in range(15, n_tr):
            tr_sum = tr_sum - tr[i-14] + tr[i]
            vm_plus_sum = vm_plus_sum - vm_plus[i-14] + vm_plus[i]
            vm_minus_sum = vm_minus_sum - vm_minus[i-14] + vm_minus[i]
            
            if tr_sum > 0:
                vi_plus[i] = vm_plus_sum / tr_sum
                vi_minus[i] = vm_minus_sum / tr_sum
    
    # Volume spike filter: current volume / 20-period average volume
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid] = volume[valid] / vol_ma[valid]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(14, 20, 50)  # Ensure VI, volume MA and EMA are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vi_plus[i]) or np.isnan(vi_minus[i]) or
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: VI+ > VI- (bullish trend) AND uptrend (price > EMA50) AND volume spike
            if (vi_plus[i] > vi_minus[i] and 
                close[i] > ema_50_12h_aligned[i] and 
                volume_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # Enter short: VI- > VI+ (bearish trend) AND downtrend (price < EMA50) AND volume spike
            elif (vi_minus[i] > vi_plus[i] and 
                  close[i] < ema_50_12h_aligned[i] and 
                  volume_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: VI- > VI+ (trend reversal) OR trend reversal (price < EMA50)
            if vi_minus[i] > vi_plus[i] or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: VI+ > VI- (trend reversal) OR trend reversal (price > EMA50)
            if vi_plus[i] > vi_minus[i] or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals