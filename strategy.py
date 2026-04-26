#!/usr/bin/env python3
"""
1h_Camarilla_R3_S3_Breakout_4hTrend_VolumeSpike
Hypothesis: Camarilla R3/S3 breakout with 4h EMA20 trend filter and volume spike confirmation on 1h timeframe.
Only trade in direction of 4h trend (EMA20) with volume > 1.5x 20-period median volume. Uses discrete sizing 0.20 to target 15-37 trades/year.
Works in bull/bear via 4h trend filter and avoids choppy markets via volume confirmation.
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
    
    # Load 4h data ONCE before loop for HTF trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h EMA20 for trend filter
    close_4h = df_4h['close'].values
    close_4h_series = pd.Series(close_4h)
    ema_20_4h = close_4h_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Calculate Camarilla levels from prior 4h bar
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla R3, S3, R4, S4 for each 4h bar
    # R4 = close + 1.5*(high-low), R3 = close + 1.125*(high-low)
    # S3 = close - 1.125*(high-low), S4 = close - 1.5*(high-low)
    rng = high_4h - low_4h
    r4 = close_4h + 1.5 * rng
    r3 = close_4h + 1.125 * rng
    s3 = close_4h - 1.125 * rng
    s4 = close_4h - 1.5 * rng
    
    # Align to 1h timeframe (wait for 4h bar to close)
    r3_1h = align_htf_to_ltf(prices, df_4h, r3)
    s3_1h = align_htf_to_ltf(prices, df_4h, s3)
    r4_1h = align_htf_to_ltf(prices, df_4h, r4)
    s4_1h = align_htf_to_ltf(prices, df_4h, s4)
    
    # Volume spike: volume > 1.5x 20-period median volume
    volume_series = pd.Series(volume)
    vol_median_20 = volume_series.rolling(window=20, min_periods=20).median().values
    volume_spike = volume > (1.5 * vol_median_20)
    
    # Fixed position size to control trade frequency
    fixed_size = 0.20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need 20 for 4h EMA, 20 for volume median
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_20_4h_aligned[i]) or
            np.isnan(r3_1h[i]) or
            np.isnan(s3_1h[i]) or
            np.isnan(r4_1h[i]) or
            np.isnan(s4_1h[i]) or
            np.isnan(vol_median_20[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_20_val = ema_20_4h_aligned[i]
        vol_spike = volume_spike[i]
        size = fixed_size
        
        if position == 0:
            # Flat - look for entry
            # Long: price > R3 and volume spike, in uptrend (close > EMA20)
            long_entry = (close_val > r3_1h[i]) and vol_spike and (close_val > ema_20_val)
            # Short: price < S3 and volume spike, in downtrend (close < EMA20)
            short_entry = (close_val < s3_1h[i]) and vol_spike and (close_val < ema_20_val)
            
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
            if close_val < ema_20_val or close_val > r4_1h[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on trend reversal or at S4 (take profit)
            if close_val > ema_20_val or close_val < s4_1h[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "1h_Camarilla_R3_S3_Breakout_4hTrend_VolumeSpike"
timeframe = "1h"
leverage = 1.0