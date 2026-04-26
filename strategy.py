#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike
Hypothesis: Camarilla R3/S3 breakout on 12h with 1w EMA34 trend filter and volume spike confirmation. 
Only trade in direction of 1w trend. Uses discrete sizing 0.25 to target 12-37 trades/year.
Designed for both bull and bear markets via 1w trend filter and volume confirmation to avoid false breakouts.
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
    
    # Load 1w data ONCE before loop for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    close_1w = df_1w['close'].values
    close_1w_series = pd.Series(close_1w)
    ema_34_1w = close_1w_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Camarilla levels from prior 1w bar
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Camarilla R3, S3, R4, S4 for each 1w bar
    rng = high_1w - low_1w
    r3 = close_1w + 1.125 * rng
    s3 = close_1w - 1.125 * rng
    r4 = close_1w + 1.5 * rng
    s4 = close_1w - 1.5 * rng
    
    # Align to 12h timeframe (wait for 1w bar to close)
    r3_12h = align_htf_to_ltf(prices, df_1w, r3)
    s3_12h = align_htf_to_ltf(prices, df_1w, s3)
    r4_12h = align_htf_to_ltf(prices, df_1w, r4)
    s4_12h = align_htf_to_ltf(prices, df_1w, s4)
    
    # Volume spike: volume > 1.5x 20-period median volume
    volume_series = pd.Series(volume)
    vol_median_20 = volume_series.rolling(window=20, min_periods=20).median().values
    volume_spike = volume > (1.5 * vol_median_20)
    
    # Fixed position size to control trade frequency
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need 34 for 1w EMA, 20 for volume median
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(r3_12h[i]) or
            np.isnan(s3_12h[i]) or
            np.isnan(r4_12h[i]) or
            np.isnan(s4_12h[i]) or
            np.isnan(vol_median_20[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_34_val = ema_34_1w_aligned[i]
        vol_spike = volume_spike[i]
        size = fixed_size
        
        if position == 0:
            # Flat - look for entry
            # Long: price > R3 and volume spike, in uptrend (close > EMA34)
            long_entry = (close_val > r3_12h[i]) and vol_spike and (close_val > ema_34_val)
            # Short: price < S3 and volume spike, in downtrend (close < EMA34)
            short_entry = (close_val < s3_12h[i]) and vol_spike and (close_val < ema_34_val)
            
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
            # Long - exit on trend reversal or at R4 (take profit)
            if close_val < ema_34_val or close_val > r4_12h[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on trend reversal or at S4 (take profit)
            if close_val > ema_34_val or close_val < s4_12h[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0