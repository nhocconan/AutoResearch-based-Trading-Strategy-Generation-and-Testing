#!/usr/bin/env python3
# 4h_1d_Vortex_Volume_Confirmation
# Hypothesis: Vortex Indicator identifies trend direction on 1d timeframe, with entry on 4h when price pulls back to EMA21 in direction of trend, confirmed by volume spike. Works in bull/bear via 1d trend filter. Target: 20-40 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Vortex_Volume_Confirmation"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # === 1d: Vortex Indicator (VI) for trend direction ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # VM+ and VM-
    vm_plus = np.abs(high_1d[1:] - low_1d[:-1])
    vm_minus = np.abs(low_1d[1:] - high_1d[:-1])
    vm_plus = np.concatenate([[np.nan], vm_plus])
    vm_minus = np.concatenate([[np.nan], vm_minus])
    
    # Smooth over 14 periods
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    vm_plus14 = pd.Series(vm_plus).rolling(window=14, min_periods=14).sum().values
    vm_minus14 = pd.Series(vm_minus).rolling(window=14, min_periods=14).sum().values
    
    # VI+ and VI-
    vi_plus = vm_plus14 / tr14
    vi_minus = vm_minus14 / tr14
    
    # Trend: VI+ > VI- = uptrend, VI- > VI+ = downtrend
    vi_plus_vi_minus = vi_plus - vi_minus  # >0 = uptrend, <0 = downtrend
    
    # Align Vortex trend to 4h
    vi_trend_aligned = align_htf_to_ltf(prices, df_1d, vi_plus_vi_minus)
    
    # === 4h: EMA21 for pullback entries ===
    close = prices['close'].values
    ema21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # === 4h: Volume ratio (current vs 20-period average) ===
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(21, n):  # Start after EMA21 warmup
        # Get values
        close_val = close[i]
        ema21_val = ema21[i]
        vi_trend_val = vi_trend_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(ema21_val) or np.isnan(vi_trend_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Uptrend (VI+ > VI-) + price at EMA21 (pullback) + volume confirmation
            if (vi_trend_val > 0 and  # 1d uptrend
                abs(close_val - ema21_val) / ema21_val < 0.005 and  # Within 0.5% of EMA21
                vol_ratio_val > 2.0):  # Volume confirmation
                signals[i] = 0.25
                position = 1
            # Short: Downtrend (VI- > VI+) + price at EMA21 (pullback) + volume confirmation
            elif (vi_trend_val < 0 and  # 1d downtrend
                  abs(close_val - ema21_val) / ema21_val < 0.005 and  # Within 0.5% of EMA21
                  vol_ratio_val > 2.0):  # Volume confirmation
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Trend reversal or price extends too far from EMA
            if (vi_trend_val < 0 or  # Trend turned down
                close_val > ema21_val * 1.02):  # Extended >2% above EMA
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Trend reversal or price extends too far from EMA
            if (vi_trend_val > 0 or  # Trend turned up
                close_val < ema21_val * 0.98):  # Extended >2% below EMA
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals