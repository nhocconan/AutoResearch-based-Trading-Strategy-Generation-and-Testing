#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike
Hypothesis: Camarilla R1/S1 breakout with 1d EMA50 trend filter and volume spike confirmation on 4h timeframe.
Only trade in direction of 1d trend (EMA50) with volume > 1.8x 24-period median volume. Uses discrete sizing 0.25 to target 20-50 trades/year.
Works in bull/bear via 1d trend filter and volume confirmation reduces false breakouts.
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
    
    # Load 1d data ONCE before loop for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    close_1d_series = pd.Series(close_1d)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Camarilla levels from prior 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla R1, S1, R2, S2 for each 1d bar
    # R2 = close + 1.25*(high-low), R1 = close + 1.083*(high-low)
    # S1 = close - 1.083*(high-low), S2 = close - 1.25*(high-low)
    rng = high_1d - low_1d
    r2 = close_1d + 1.25 * rng
    r1 = close_1d + 1.083333 * rng
    s1 = close_1d - 1.083333 * rng
    s2 = close_1d - 1.25 * rng
    
    # Align to 4h timeframe (wait for 1d bar to close)
    r1_4h = align_htf_to_ltf(prices, df_1d, r1)
    s1_4h = align_htf_to_ltf(prices, df_1d, s1)
    r2_4h = align_htf_to_ltf(prices, df_1d, r2)
    s2_4h = align_htf_to_ltf(prices, df_1d, s2)
    
    # Volume spike: volume > 1.8x 24-period median volume
    volume_series = pd.Series(volume)
    vol_median_24 = volume_series.rolling(window=24, min_periods=24).median().values
    volume_spike = volume > (1.8 * vol_median_24)
    
    # Fixed position size to control trade frequency
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need 50 for 1d EMA, 24 for volume median
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(r1_4h[i]) or
            np.isnan(s1_4h[i]) or
            np.isnan(r2_4h[i]) or
            np.isnan(s2_4h[i]) or
            np.isnan(vol_median_24[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_50_val = ema_50_1d_aligned[i]
        vol_spike = volume_spike[i]
        size = fixed_size
        
        if position == 0:
            # Flat - look for entry
            # Long: price > R1 and volume spike, in uptrend (close > EMA50)
            long_entry = (close_val > r1_4h[i]) and vol_spike and (close_val > ema_50_val)
            # Short: price < S1 and volume spike, in downtrend (close < EMA50)
            short_entry = (close_val < s1_4h[i]) and vol_spike and (close_val < ema_50_val)
            
            if long_entry:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_entry:
                signals[i] = -size
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on trend reversal or at R2 (take profit)
            if close_val < ema_50_val or close_val > r2_4h[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on trend reversal or at S2 (take profit)
            if close_val > ema_50_val or close_val < s2_4h[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0