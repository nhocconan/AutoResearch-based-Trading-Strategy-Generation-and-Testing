#!/usr/bin/env python3
"""
12h Vortex Trend + Volume Spike
Hypothesis: Vortex Indicator identifies trend direction (VI+ > VI- for uptrend, VI- > VI+ for downtrend). 
Combined with volume spikes (>2x 20-period average) to confirm institutional participation. 
Works in bull markets via VI+ crossovers and in bear markets via VI- crossovers. 
Low trade frequency due to requiring both trend and volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_vortex(high, low, close):
    """Calculate Vortex Indicator: VI+ and VI-"""
    tr = np.maximum(high - low, np.maximum(abs(high - np.roll(close, 1)), abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]  # First TR
    
    vm_plus = abs(high - np.roll(low, 1))
    vm_minus = abs(low - np.roll(high, 1))
    
    # Sum over period (default 14)
    period = 14
    vi_plus = pd.Series(vm_plus).rolling(window=period, min_periods=period).sum() / pd.Series(tr).rolling(window=period, min_periods=period).sum()
    vi_minus = pd.Series(vm_minus).rolling(window=period, min_periods=period).sum() / pd.Series(tr).rolling(window=period, min_periods=period).sum()
    
    return vi_plus.values, vi_minus.values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Vortex (once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Vortex on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    vi_plus_1d, vi_minus_1d = calculate_vortex(high_1d, low_1d, close_1d)
    
    # Align 1d Vortex to 12h timeframe
    vi_plus_1d_aligned = align_htf_to_ltf(prices, df_1d, vi_plus_1d)
    vi_minus_1d_aligned = align_htf_to_ltf(prices, df_1d, vi_minus_1d)
    
    # Volume confirmation: current volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Warmup for indicators
    
    for i in range(start_idx, n):
        if np.isnan(vi_plus_1d_aligned[i]) or np.isnan(vi_minus_1d_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        vi_plus = vi_plus_1d_aligned[i]
        vi_minus = vi_minus_1d_aligned[i]
        vol_ok = vol_confirm[i]
        
        if position == 0:
            # Enter long: VI+ crosses above VI- with volume confirmation
            if vi_plus > vi_minus and vol_ok:
                signals[i] = 0.25
                position = 1
            # Enter short: VI- crosses above VI+ with volume confirmation
            elif vi_minus > vi_plus and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: VI- crosses above VI+ (trend change)
            if vi_minus > vi_plus:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: VI+ crosses above VI- (trend change)
            if vi_plus > vi_minus:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Vortex_Trend_Volume_Spike"
timeframe = "12h"
leverage = 1.0