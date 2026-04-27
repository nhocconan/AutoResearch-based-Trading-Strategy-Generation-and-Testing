#!/usr/bin/env python3
"""
4h_Vortex_Signal_1dTrend_Volume
Hypothesis: Vortex Indicator identifies trend direction and strength; combined with 1d EMA50 trend filter and volume spike confirmation to capture strong trends while avoiding whipsaws. Works in bull markets via strong VI+ signals and in bear markets via strong VI- signals. Targets ~20-30 trades/year on 4h to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 80:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Vortex Indicator (14-period) on 4h data
    # VM+ = |High - Prior Low|, VM- = |Low - Prior High|
    vm_plus = np.abs(high - np.roll(low, 1))
    vm_minus = np.abs(low - np.roll(high, 1))
    vm_plus[0] = np.abs(high[0] - low[0])  # first value
    vm_minus[0] = np.abs(low[0] - high[0])  # first value
    
    # Sum of true range
    tr1 = np.abs(high - np.roll(low, 1))
    tr2 = np.abs(low - np.roll(high, 1))
    tr3 = np.abs(high - np.roll(high, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = high[0] - low[0]  # first true range
    
    # Smooth over 14 periods
    vi_plus = pd.Series(vm_plus).rolling(window=14, min_periods=14).sum().values / \
              pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    vi_minus = pd.Series(vm_minus).rolling(window=14, min_periods=14).sum().values / \
               pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Volume confirmation: volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for Vortex, EMA, and volume MA
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(vi_plus[i]) or np.isnan(vi_minus[i]) or 
            np.isnan(ema50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        vi_p = vi_plus[i]
        vi_m = vi_minus[i]
        ema_trend = ema50_1d_aligned[i]
        vol_spike_val = vol_spike[i]
        
        if position == 0:
            # Long: VI+ > VI- (bullish trend) with uptrend and volume spike
            if vi_p > vi_m and vol_spike_val and close[i] > ema_trend:
                signals[i] = size
                position = 1
            # Short: VI- > VI+ (bearish trend) with downtrend and volume spike
            elif vi_m > vi_p and vol_spike_val and close[i] < ema_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: trend weakens (VI- > VI+) or trend turns down
            if vi_m > vi_p or close[i] < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: trend weakens (VI+ > VI-) or trend turns up
            if vi_p > vi_m or close[i] > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Vortex_Signal_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0