#!/usr/bin/env python3
"""
4h_Vortex_VolumeSpike_1dTrend
Hypothesis: Vortex Indicator (VI+) and (VI-) identify trend direction, with VI+ > VI- for uptrend and VI- > VI+ for downtrend. 
Add 1d EMA34 as higher timeframe trend filter and volume spike confirmation to avoid false signals. 
Designed for 20-50 trades per year on 4h timeframe, works in bull via VI+ > VI- above EMA34, bear via VI- > VI+ below EMA34.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Vortex Indicator on 4h
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # tr[0] = nan
    
    # VM+ and VM-
    vm_plus = np.abs(high[1:] - low[:-1])
    vm_minus = np.abs(low[1:] - high[:-1])
    vm_plus = np.concatenate([[np.nan], vm_plus])
    vm_minus = np.concatenate([[np.nan], vm_minus])
    
    # Sum over 14 periods
    period = 14
    sum_tr = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    sum_vm_plus = pd.Series(vm_plus).rolling(window=period, min_periods=period).sum().values
    sum_vm_minus = pd.Series(vm_minus).rolling(window=period, min_periods=period).sum().values
    
    # VI+ and VI-
    vi_plus = sum_vm_plus / sum_tr
    vi_minus = sum_vm_minus / sum_tr
    
    # Volume spike: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for VI calculations
    start_idx = period + 20  # 14 + 20 = 34
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(vi_plus[i]) or np.isnan(vi_minus[i]) or 
            np.isnan(ema_34_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        vi_plus_val = vi_plus[i]
        vi_minus_val = vi_minus[i]
        ema_34_val = ema_34_1d_aligned[i]
        vol_spike_val = vol_spike[i]
        
        if position == 0:
            # Long: VI+ > VI- AND price above EMA34 AND volume spike
            if vi_plus_val > vi_minus_val and close[i] > ema_34_val and vol_spike_val:
                signals[i] = size
                position = 1
            # Short: VI- > VI+ AND price below EMA34 AND volume spike
            elif vi_minus_val > vi_plus_val and close[i] < ema_34_val and vol_spike_val:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: VI- > VI+ (trend change) OR price drops below EMA34
            if vi_minus_val > vi_plus_val or close[i] < ema_34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: VI+ > VI- (trend change) OR price rises above EMA34
            if vi_plus_val > vi_minus_val or close[i] > ema_34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Vortex_VolumeSpike_1dTrend"
timeframe = "4h"
leverage = 1.0