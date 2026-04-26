#!/usr/bin/env python3
"""
4h_Vortex_Trend_VolumeSpike_v1
Hypothesis: Vortex indicator (VI+ > VI-) identifies trend direction, volume spike (>2x 48-bar average) confirms institutional participation, and ATR-based trailing stop (3x ATR) manages risk. Works in bull/bear by capturing strong trending moves with volume confirmation. Designed for 4h to target 20-50 trades/year with discrete sizing (0.30).
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
    
    # ATR(14) for volatility and trailing stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Vortex indicator (VI+ and VI-) over 14 periods
    vm_plus = np.abs(high - np.roll(low, 1))
    vm_minus = np.abs(low - np.roll(high, 1))
    vm_plus[0] = 0
    vm_minus[0] = 0
    sum_vm_plus = pd.Series(vm_plus).rolling(window=14, min_periods=14).sum().values
    sum_vm_minus = pd.Series(vm_minus).rolling(window=14, min_periods=14).sum().values
    sum_tr = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    vi_plus = sum_vm_plus / sum_tr
    vi_minus = sum_vm_minus / sum_tr
    
    # Average volume for confirmation (48-period SMA = 2 days * 12 bars/day)
    avg_volume = pd.Series(volume).rolling(window=48, min_periods=48).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_long = 0.0
    lowest_since_short = 0.0
    base_size = 0.30
    
    # Warmup: max of Vortex(14), ATR(14), volume(48)
    start_idx = max(14, 14, 48)
    
    for i in range(start_idx, n):
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        vi_plus_val = vi_plus[i]
        vi_minus_val = vi_minus[i]
        atr_val = atr[i]
        
        # Skip if any data not ready
        if (np.isnan(vi_plus_val) or np.isnan(vi_minus_val) or 
            np.isnan(avg_vol) or np.isnan(atr_val)):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Volume confirmation: current volume > 2.0x average volume
        volume_confirmed = vol > 2.0 * avg_vol
        
        # Vortex trend: VI+ > VI- = uptrend, VI- > VI+ = downtrend
        uptrend = vi_plus_val > vi_minus_val
        downtrend = vi_minus_val > vi_plus_val
        
        # Update highest/lowest since position entry
        if position == 1:
            highest_since_long = max(highest_since_long, high_val)
        elif position == -1:
            lowest_since_short = min(lowest_since_short, low_val)
        
        # Long: VI+ crosses above VI- with volume confirmation
        long_cross = (vi_plus_val > vi_minus_val) and (np.isnan(vi_plus[i-1]) or vi_plus[i-1] <= vi_minus[i-1])
        long_condition = long_cross and volume_confirmed
        # Short: VI- crosses above VI+ with volume confirmation
        short_cross = (vi_minus_val > vi_plus_val) and (np.isnan(vi_minus[i-1]) or vi_minus[i-1] <= vi_plus[i-1])
        short_condition = short_cross and volume_confirmed
        
        # Exit: price retraces 3x ATR from extreme
        long_exit = (position == 1 and close_val <= highest_since_long - 3.0 * atr_val)
        short_exit = (position == -1 and close_val >= lowest_since_short + 3.0 * atr_val)
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
            entry_price = close_val
            highest_since_long = high_val
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
            entry_price = close_val
            lowest_since_short = low_val
        elif long_exit:
            signals[i] = 0.0
            position = 0
            highest_since_long = 0.0
        elif short_exit:
            signals[i] = 0.0
            position = 0
            lowest_since_short = 0.0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "4h_Vortex_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0