#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_Vortex_VolumeTrendFilter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 36:  # Need at least 3 days of 12h data for 20-period lookback
        return np.zeros(n)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # === 1d: Vortex Indicator (14-period) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Vortex Movement
    vm_plus = np.abs(high_1d[1:] - low_1d[:-1])
    vm_minus = np.abs(low_1d[1:] - high_1d[:-1])
    vm_plus = np.concatenate([[np.nan], vm_plus])
    vm_minus = np.concatenate([[np.nan], vm_minus])
    
    # Sum over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    vm_plus_sum = pd.Series(vm_plus).rolling(window=14, min_periods=14).sum().values
    vm_minus_sum = pd.Series(vm_minus).rolling(window=14, min_periods=14).sum().values
    
    # VI+ and VI-
    vi_plus = vm_plus_sum / tr_sum
    vi_minus = vm_minus_sum / tr_sum
    
    # === 12h: EMA25 for trend filter ===
    close = prices['close'].values
    ema_25 = pd.Series(close).ewm(span=25, adjust=False, min_periods=25).mean().values
    
    # === 12h: Volume ratio (current vs 20-period average) ===
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # Align 1d Vortex to 12h timeframe
    vi_plus_aligned = align_htf_to_ltf(prices, df_1d, vi_plus)
    vi_minus_aligned = align_htf_to_ltf(prices, df_1d, vi_minus)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):  # Start after Vortex warmup
        # Get values
        close_val = close[i]
        ema_val = ema_25[i]
        vi_plus_val = vi_plus_aligned[i]
        vi_minus_val = vi_minus_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(ema_val) or np.isnan(vi_plus_val) or np.isnan(vi_minus_val) or
            np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: VI+ > VI- (bullish), price above EMA (uptrend), volume confirmation
            if (vi_plus_val > vi_minus_val and  # Bullish Vortex
                close_val > ema_val and         # Uptrend filter
                vol_ratio_val > 1.5):           # Volume confirmation
                signals[i] = 0.25
                position = 1
            # Short: VI- > VI+ (bearish), price below EMA (downtrend), volume confirmation
            elif (vi_minus_val > vi_plus_val and  # Bearish Vortex
                  close_val < ema_val and         # Downtrend filter
                  vol_ratio_val > 1.5):           # Volume confirmation
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: VI- crosses above VI+ (trend change) or price breaks below EMA
            if vi_minus_val > vi_plus_val or close_val < ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: VI+ crosses above VI- (trend change) or price breaks above EMA
            if vi_plus_val > vi_minus_val or close_val > ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals