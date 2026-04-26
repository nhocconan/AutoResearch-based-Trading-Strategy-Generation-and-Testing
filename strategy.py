#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_12hTrend_VolumeSpike_Dyn_v1
Hypothesis: Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume spike confirmation. 
Only long when price > EMA50(12h), short when price < EMA50(12h). Uses dynamic position sizing 
based on ATR volatility (0.25 in low vol, 0.35 in high vol). Targets 100-180 total trades over 4 years.
Designed to work in both bull (breakouts with trend) and bear (breakouts against trend filtered by EMA) regimes.
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
    
    # Calculate Camarilla levels from previous day
    # Using daily data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Typical price for Camarilla calculation
    typical_price = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels: R3, R2, R1, PP, S1, S2, S3
    # R4 = close + range * 1.1/2, R3 = close + range * 1.1/4, etc.
    camarilla_multiplier = 1.1 / 4
    r3 = close_1d + range_1d * camarilla_multiplier * 3
    r2 = close_1d + range_1d * camarilla_multiplier * 2
    r1 = close_1d + range_1d * camarilla_multiplier
    pp = typical_price
    s1 = close_1d - range_1d * camarilla_multiplier
    s2 = close_1d - range_1d * camarilla_multiplier * 2
    s3 = close_1d - range_1d * camarilla_multiplier * 3
    
    # Align Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Load 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # ATR for volatility-based position sizing
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume confirmation: volume > 1.5 * 20-period EMA volume
    avg_volume = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(50, 20, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(atr[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Dynamic position sizing based on ATR volatility
        atr_ratio = atr[i] / (np.mean(atr[max(0, i-50):i+1]) + 1e-10)
        if atr_ratio > 1.2:  # High volatility
            base_size = 0.35
        elif atr_ratio < 0.8:  # Low volatility
            base_size = 0.25
        else:  # Normal volatility
            base_size = 0.30
        
        # Long logic: price breaks above R3 with volume spike and above 12h EMA50
        if close[i] > r3_aligned[i] and volume_spike[i] and close[i] > ema_50_12h_aligned[i]:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: price breaks below S3 with volume spike and below 12h EMA50
        elif close[i] < s3_aligned[i] and volume_spike[i] and close[i] < ema_50_12h_aligned[i]:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit conditions: price returns to pivot point or opposite breakout
        elif position == 1 and (close[i] < pp_aligned[i] or close[i] < s3_aligned[i]):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close[i] > pp_aligned[i] or close[i] > r3_aligned[i]):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_12hTrend_VolumeSpike_Dyn_v1"
timeframe = "4h"
leverage = 1.0