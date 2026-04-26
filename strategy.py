#!/usr/bin/env python3
"""
1h_Camarilla_R3_S3_Breakout_4hTrend_1dVolumeConfirm_v1
Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and 1d volume spike confirmation.
Only long when price > 4h EMA50, short when price < 4h EMA50. Volume confirmation from 1d ensures institutional participation.
Designed for 1h timeframe with tight entries (target: 60-150 trades over 4 years) to avoid fee drag.
Uses session filter (08-20 UTC) to reduce noise. Fixed position size 0.20 to minimize churn.
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
    
    # Calculate Camarilla levels from previous 1h bar (using 1h data for pivot)
    # Camarilla uses typical price and range from previous bar
    df_1h = get_htf_data(prices, '1h')
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    close_1h = df_1h['close'].values
    
    typical_price = (high_1h + low_1h + close_1h) / 3
    range_1h = high_1h - low_1h
    
    camarilla_multiplier = 1.1 / 4
    r3 = close_1h + range_1h * camarilla_multiplier * 3
    r2 = close_1h + range_1h * camarilla_multiplier * 2
    r1 = close_1h + range_1h * camarilla_multiplier
    pp = typical_price
    s1 = close_1h - range_1h * camarilla_multiplier
    s2 = close_1h - range_1h * camarilla_multiplier * 2
    s3 = close_1h - range_1h * camarilla_multiplier * 3
    
    # Align Camarilla levels to 1h timeframe (already 1h, but using helper for consistency)
    r3_aligned = align_htf_to_ltf(prices, df_1h, r3)
    r2_aligned = align_htf_to_ltf(prices, df_1h, r2)
    r1_aligned = align_htf_to_ltf(prices, df_1h, r1)
    pp_aligned = align_htf_to_ltf(prices, df_1h, pp)
    s1_aligned = align_htf_to_ltf(prices, df_1h, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1h, s2)
    s3_aligned = align_htf_to_ltf(prices, df_1h, s3)
    
    # Load 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Load 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    avg_volume_1d = pd.Series(volume_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (1.5 * avg_volume_1d)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(50, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready or outside session
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(volume_spike_1d_aligned[i]) or
            not in_session[i]):
            # Hold current position
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Long logic: price breaks above R3 with 1d volume spike and above 4h EMA50
        if close[i] > r3_aligned[i] and volume_spike_1d_aligned[i] and close[i] > ema_50_4h_aligned[i]:
            if position != 1:
                signals[i] = 0.20
                position = 1
            else:
                signals[i] = 0.20
        # Short logic: price breaks below S3 with 1d volume spike and below 4h EMA50
        elif close[i] < s3_aligned[i] and volume_spike_1d_aligned[i] and close[i] < ema_50_4h_aligned[i]:
            if position != -1:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = -0.20
        # Exit conditions: price returns to pivot point
        elif position == 1 and close[i] < pp_aligned[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > pp_aligned[i]:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
    
    return signals

name = "1h_Camarilla_R3_S3_Breakout_4hTrend_1dVolumeConfirm_v1"
timeframe = "1h"
leverage = 1.0