#!/usr/bin/env python3
"""
4h_DonchianBreakout_VolumeTrend_v1
Hypothesis: Donchian(20) breakout with volume confirmation and trend filter (1d EMA50) captures strong trends while avoiding false breakouts in choppy markets. Works in both bull and bear by capturing momentum bursts. Target: 20-50 trades/year.
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
    
    # === Donchian Channels (20) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === Volume Confirmation (volume > 1.5x 20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    
    # === 1d EMA50 for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(vol_ratio[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price breaks above Donchian high, volume confirmation, above 1d EMA50
            if (close[i] > donchian_high[i-1] and  # breakout confirmed on close
                vol_ratio[i] > 1.5 and
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below Donchian low, volume confirmation, below 1d EMA50
            elif (close[i] < donchian_low[i-1] and  # breakdown confirmed on close
                  vol_ratio[i] > 1.5 and
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price re-enters Donchian channel (below midpoint) OR volume drops
            midpoint = (donchian_high[i] + donchian_low[i]) / 2.0
            if (close[i] < midpoint or 
                vol_ratio[i] < 1.0):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price re-enters Donchian channel (above midpoint) OR volume drops
            midpoint = (donchian_high[i] + donchian_low[i]) / 2.0
            if (close[i] > midpoint or 
                vol_ratio[i] < 1.0):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_DonchianBreakout_VolumeTrend_v1"
timeframe = "4h"
leverage = 1.0