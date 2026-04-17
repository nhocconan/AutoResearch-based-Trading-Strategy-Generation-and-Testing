#!/usr/bin/env python3
"""
4h_CCI_Overbought_Oversold_R1S1_v1
Long: CCI(20) < -100 + price touches S1 pivot from 1d + volume spike
Short: CCI(20) > 100 + price touches R1 pivot from 1d + volume spike
Exit: CCI crosses back within [-50, 50] OR price reaches opposite pivot (R2/S2)
Designed to capture mean-reversion bounces at key levels with volume confirmation.
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_hlf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === CCI(20) ===
    typical_price = (high + low + close) / 3.0
    sma_tp = pd.Series(typical_price).rolling(window=20, min_periods=20).mean().values
    mad = pd.Series(typical_price).rolling(window=20, min_periods=20).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    ).values
    cci = (typical_price - sma_tp) / (0.015 * mad)
    
    # === 1d Pivot Points (Classic) ===
    df_1d = get_htf_data(prices, '1d')
    # Calculate pivot points from previous day's OHLC
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Pivot point and support/resistance levels
    pp = (prev_high + prev_low + prev_close) / 3.0
    r1 = 2 * pp - prev_low
    s1 = 2 * pp - prev_high
    r2 = pp + (prev_high - prev_low)
    s2 = pp - (prev_high - prev_low)
    
    # Align to 4h timeframe (use previous day's levels)
    pp_aligned = align_ltf_to_hlf(prices, df_1d, pp)
    r1_aligned = align_ltf_to_hlf(prices, df_1d, r1)
    s1_aligned = align_ltf_to_hlf(prices, df_1d, s1)
    r2_aligned = align_ltf_to_hlf(prices, df_1d, r2)
    s2_aligned = align_ltf_to_hlf(prices, df_1d, s2)
    
    # === Volume Spike (2x 20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2 * vol_ma)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(cci[i]) or 
            np.isnan(pp_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(r2_aligned[i]) or 
            np.isnan(s2_aligned[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: CCI < -100 (oversold), price touches S1, volume spike
            if (cci[i] < -100 and 
                low[i] <= s1_aligned[i] * 1.001 and  # Allow small tolerance for touch
                high[i] >= s1_aligned[i] * 0.999 and
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: CCI > 100 (overbought), price touches R1, volume spike
            elif (cci[i] > 100 and 
                  high[i] >= r1_aligned[i] * 0.999 and  # Allow small tolerance for touch
                  low[i] <= r1_aligned[i] * 1.001 and
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: CCI > -50 OR price reaches R2
            if (cci[i] > -50 or 
                high[i] >= r2_aligned[i] * 0.999):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: CCI < 50 OR price reaches S2
            if (cci[i] < 50 or 
                low[i] <= s2_aligned[i] * 1.001):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_CCI_Overbought_Oversold_R1S1_v1"
timeframe = "4h"
leverage = 1.0