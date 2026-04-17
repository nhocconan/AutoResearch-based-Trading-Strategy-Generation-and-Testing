#!/usr/bin/env python3
"""
12h_KAMA_Slope_Trend_v1
KAMA trend + 1d Donchian breakout + volume filter.
Long: KAMA slope up + close > 1d high + volume spike.
Short: KAMA slope down + close < 1d low + volume spike.
Exit when KAMA reverses or volume drops.
Designed for low-turnover trend capture with volume confirmation.
Target: 15-30 trades/year (60-120 total over 4 years).
"""

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
    
    # === KAMA (ER=10, SC=2,30) ===
    change = np.abs(close - np.roll(close, 10))
    change[0:10] = 0  # first 10 bars: change = close - close[0] but we'll handle via volatility
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0)  # not quite right, need rolling
    # Recompute properly:
    er = np.zeros(n)
    for i in range(10, n):
        if i >= 10:
            change_val = np.abs(close[i] - close[i-10])
            volatility_sum = np.sum(np.abs(np.diff(close[i-9:i+1])))
            er[i] = change_val / (volatility_sum + 1e-10)
    er[0:10] = 0
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === 1d Donchian channels (20) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    for i in range(20, len(high_1d)):
        donch_high[i] = np.max(high_1d[i-20:i])
        donch_low[i] = np.min(low_1d[i-20:i])
    
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    
    # === Volume spike (volume > 1.5 * 20-period MA) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or i < 10 or  # KAMA needs lookback
            np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # KAMA slope (current - previous)
        kama_slope = kama[i] - kama[i-1]
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: KAMA slope up + close > 1d Donchian high + volume spike
            if (kama_slope > 0 and 
                close[i] > donch_high_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: KAMA slope down + close < 1d Donchian low + volume spike
            elif (kama_slope < 0 and 
                  close[i] < donch_low_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: KAMA slope down OR no volume spike
            if (kama_slope < 0 or 
                not vol_spike[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: KAMA slope up OR no volume spike
            if (kama_slope > 0 or 
                not vol_spike[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_KAMA_Slope_Trend_v1"
timeframe = "12h"
leverage = 1.0