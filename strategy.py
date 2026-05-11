#!/usr/bin/env python3
"""
4h_Vortex_VolumeSpike_Trend_v1
Hypothesis: Vortex Indicator (VI) detects trend direction, with VI+ > VI- indicating uptrend and VI- > VI+ indicating downtrend. Combined with volume spike confirmation and price position relative to 1-day EMA50 for trend strength. Works in bull markets (VI+ cross above VI- in uptrend) and bear markets (VI- cross above VI+ in downtrend). Volume spike confirms institutional interest. Target: 20-50 trades per year on 4h timeframe.
"""

name = "4h_Vortex_VolumeSpike_Trend_v1"
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
    
    # === 1D Data for Trend Filter and Vortex Calculation ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for Vortex calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First day
    
    # Vortex Indicator components
    vm_plus = np.abs(high_1d - np.roll(low_1d, 1))
    vm_minus = np.abs(low_1d - np.roll(high_1d, 1))
    vm_plus[0] = np.abs(high_1d[0] - low_1d[0])  # First day
    vm_minus[0] = np.abs(low_1d[0] - high_1d[0])  # First day
    
    # Vortex Indicator (VI) - 14 period
    period = 14
    sum_vm_plus = pd.Series(vm_plus).rolling(window=period, min_periods=period).sum().values
    sum_vm_minus = pd.Series(vm_minus).rolling(window=period, min_periods=period).sum().values
    sum_tr = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    
    vi_plus = sum_vm_plus / sum_tr
    vi_minus = sum_vm_minus / sum_tr
    
    # Trend filter: EMA50 on 1d close
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1D indicators to 4h timeframe
    vi_plus_aligned = align_htf_to_ltf(prices, df_1d, vi_plus)
    vi_minus_aligned = align_htf_to_ltf(prices, df_1d, vi_minus)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume spike: current volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(vi_plus_aligned[i]) or 
            np.isnan(vi_minus_aligned[i]) or 
            np.isnan(ema_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: VI+ crosses above VI- AND price above EMA50 AND volume spike
            if (vi_plus_aligned[i] > vi_minus_aligned[i] and 
                vi_plus_aligned[i-1] <= vi_minus_aligned[i-1] and
                close[i] > ema_50_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: VI- crosses above VI+ AND price below EMA50 AND volume spike
            elif (vi_minus_aligned[i] > vi_plus_aligned[i] and 
                  vi_minus_aligned[i-1] <= vi_plus_aligned[i-1] and
                  close[i] < ema_50_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: VI- crosses above VI+ OR price crosses below EMA50
            if (vi_minus_aligned[i] > vi_plus_aligned[i] and 
                vi_minus_aligned[i-1] <= vi_plus_aligned[i-1]) or \
               close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: VI+ crosses above VI- OR price crosses above EMA50
            if (vi_plus_aligned[i] > vi_minus_aligned[i] and 
                vi_plus_aligned[i-1] <= vi_minus_aligned[i-1]) or \
               close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals