#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla pivot (R3/S3) breakout with volume confirmation and 12h EMA trend filter.
- Camarilla levels from daily data provide institutional pivot points for reversals/breakouts
- Volume spike (1.5x 20-period avg) confirms participation
- 12h EMA50 filters for trend direction to avoid counter-trend trades
- Target: 25-40 trades/year to avoid fee drag
- Uses discrete position sizing (0.25) to minimize churn
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla levels and 12h EMA
    df_1d = get_htf_data(prices, '1d')
    df_12h = get_htf_data(prices, '12h')
    if len(df_1d) < 20 or len(df_12h) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels (R3, S3)
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # R3 = Close + (Range * 1.1/2)
    # S3 = Close - (Range * 1.1/2)
    pivot = (high_1d + low_1d + close_1d) / 3.0
    rng = high_1d - low_1d
    camarilla_r3 = close_1d + (rng * 1.1 / 2.0)
    camarilla_s3 = close_1d - (rng * 1.1 / 2.0)
    
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50 = np.full(len(close_12h), np.nan)
    if len(close_12h) >= 50:
        ema_50[49] = np.mean(close_12h[:50])
        for i in range(50, len(close_12h)):
            ema_50[i] = (close_12h[i] * 0.04) + (ema_50[i-1] * 0.96)  # 2/(50+1) = 0.04
    
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # Volume spike: current volume > 1.5 * 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup
    start_idx = max(30, 50)
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema_50_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: price breaks above R3 + volume spike + price > 12h EMA50 (uptrend)
            if (close[i] > camarilla_r3_aligned[i] and 
                volume_spike[i] and 
                close[i] > ema_50_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S3 + volume spike + price < 12h EMA50 (downtrend)
            elif (close[i] < camarilla_s3_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < ema_50_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price breaks below S3 (reversal) OR volume drops
            if (close[i] < camarilla_s3_aligned[i] or 
                not volume_spike[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above R3 (reversal) OR volume drops
            if (close[i] > camarilla_r3_aligned[i] or 
                not volume_spike[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R3S3_VolumeSpike_12hEMA50_Trend_v1"
timeframe = "4h"
leverage = 1.0