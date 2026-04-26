#!/usr/bin/env python3
"""
6h_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike_v1
Hypothesis: On 6h timeframe, Camarilla R3/S3 breakouts with 1w trend filter (EMA50) and volume confirmation (>2x avg) provides robust directional signals. Works in bull markets (long when price > weekly EMA50 + R3 breakout) and bear markets (short when price < weekly EMA50 + S3 breakdown). Uses discrete sizing (0.0, ±0.25) to minimize fee churn. Targets 50-150 trades over 4 years (12-37/year) for optimal 6h frequency. Weekly trend avoids whipsaws in counter-trend breakouts.
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
    
    # Get 1w data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # need enough for EMA
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = pd.Series(df_1w['close'].values)
    ema_50_1w = close_1w.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Camarilla levels from previous day's range
    # Need daily data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's high, low, close for Camarilla
    prev_high = df_1d['high'].shift(1).values  # shift(1) for previous completed day
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Align daily data to 6h
    prev_high_6h = align_htf_to_ltf(prices, df_1d, prev_high)
    prev_low_6h = align_htf_to_ltf(prices, df_1d, prev_low)
    prev_close_6h = align_htf_to_ltf(prices, df_1d, prev_close)
    
    # Calculate Camarilla levels
    range_ = prev_high_6h - prev_low_6h
    # Avoid division by zero
    range_ = np.maximum(range_, 1e-10)
    
    # Camarilla R3, R4, S3, S4
    r3 = prev_close_6h + range_ * 1.1 / 4
    r4 = prev_close_6h + range_ * 1.1 / 2
    s3 = prev_close_6h - range_ * 1.1 / 4
    s4 = prev_close_6h - range_ * 1.1 / 2
    
    # Volume ratio (current / 20-period average) for spike confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.maximum(vol_ma, 1e-10)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need previous day data + EMA warmup + volume MA
    start_idx = max(50, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(r3[i]) or np.isnan(r4[i]) or
            np.isnan(s3[i]) or np.isnan(s4[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(prev_high_6h[i]) or np.isnan(prev_low_6h[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        vol_confirmed = vol_ratio[i] > 2.0  # volume at least 2.0x average
        
        if position == 0:
            # Long: price > weekly EMA50 + breaks above R3 + volume
            long_signal = (close[i] > ema_50_1w_aligned[i] and 
                          close[i] > r3[i] and 
                          vol_confirmed)
            
            # Short: price < weekly EMA50 + breaks below S3 + volume
            short_signal = (close[i] < ema_50_1w_aligned[i] and 
                           close[i] < s3[i] and 
                           vol_confirmed)
            
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
            # Exit: price closes below weekly EMA50 OR breaks below S3 (reversal)
            if close[i] < ema_50_1w_aligned[i] or close[i] < s3[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price closes above weekly EMA50 OR breaks above R3 (reversal)
            if close[i] > ema_50_1w_aligned[i] or close[i] > r3[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0