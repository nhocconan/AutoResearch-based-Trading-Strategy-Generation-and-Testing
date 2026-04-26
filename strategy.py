#!/usr/bin/env python3
"""
4h_Vortex_VolumeSpike_TrendRegime_v1
Hypothesis: Use Vortex Indicator (VI+) and VI- crossover for trend direction on 4h,
confirmed by 1d EMA50 trend filter and volume spike (>2.0x median). Enter long when VI+ crosses above VI- 
in uptrend, short when VI- crosses above VI+ in downtrend. Exit on opposite crossover or trend change.
Designed for 20-40 trades/year with discrete sizing (0.25) to minimize fee drag.
Works in bull/bear markets by following 1d EMA50 trend - avoids counter-trend trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Vortex Indicator on 4h (primary timeframe)
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Vortex Movements
    vm_plus = np.abs(high - np.roll(low, 1))
    vm_minus = np.abs(low - np.roll(high, 1))
    vm_plus[0] = np.nan
    vm_minus[0] = np.nan
    
    # Sum over 14 periods
    period = 14
    tr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    vm_plus_sum = pd.Series(vm_plus).rolling(window=period, min_periods=period).sum().values
    vm_minus_sum = pd.Series(vm_minus).rolling(window=period, min_periods=period).sum().values
    
    # VI+ and VI-
    vi_plus = vm_plus_sum / tr_sum
    vi_minus = vm_minus_sum / tr_sum
    
    # Align HTF indicators to 4h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of Vortex (14), EMA (50)
    start_idx = max(50, period)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(vi_plus[i]) or np.isnan(vi_minus[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            # Hold current position
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        vi_plus_val = vi_plus[i]
        vi_minus_val = vi_minus[i]
        ema_50_1d_val = ema_50_1d_aligned[i]
        close_val = close[i]
        
        if position == 0:
            # Long: VI+ crosses above VI- in uptrend (close > 1d EMA50)
            long_signal = (vi_plus_val > vi_minus_val) and (vi_plus[i-1] <= vi_minus[i-1]) and (close_val > ema_50_1d_val)
            
            # Short: VI- crosses above VI+ in downtrend (close < 1d EMA50)
            short_signal = (vi_minus_val > vi_plus_val) and (vi_minus[i-1] <= vi_plus[i-1]) and (close_val < ema_50_1d_val)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: VI- crosses above VI+ (reversal) or trend changes (close < 1d EMA50)
            if (vi_minus_val > vi_plus_val) and (vi_minus[i-1] <= vi_plus[i-1]) or \
               (close_val < ema_50_1d_val):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: VI+ crosses above VI- (reversal) or trend changes (close > 1d EMA50)
            if (vi_plus_val > vi_minus_val) and (vi_plus[i-1] <= vi_minus[i-1]) or \
               (close_val > ema_50_1d_val):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Vortex_VolumeSpike_TrendRegime_v1"
timeframe = "4h"
leverage = 1.0