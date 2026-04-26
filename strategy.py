#!/usr/bin/env python3
"""
6h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_v1
Hypothesis: Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation.
Only long when price breaks above R3 and close > 1d EMA34, short when breaks below S3 and close < 1d EMA34.
Uses discrete position sizing (0.0, ±0.25) to minimize fee churn. Target: 50-150 total trades over 4 years (12-37/year).
Works in both bull and bear markets by combining price structure (Camarilla) with trend (1d EMA) and volume filters.
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
    
    # Calculate 1d Camarilla pivot levels (based on previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_close = df_1d['close'].shift(1).values  # Previous day's close
    prev_high = df_1d['high'].shift(1).values    # Previous day's high
    prev_low = df_1d['low'].shift(1).values      # Previous day's low
    
    # Calculate Camarilla levels
    range_ = prev_high - prev_low
    camarilla_r3 = prev_close + range_ * 1.1 / 4
    camarilla_s3 = prev_close - range_ * 1.1 / 4
    
    # Align Camarilla levels to 6h timeframe (already delayed by shift(1) for previous day)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # 1d EMA34 trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 1.5 * 20-period EMA volume
    avg_volume = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need at least 1 day of previous data + EMA warmup)
    start_idx = max(34, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Discrete position sizing
        base_size = 0.25
        
        # Long logic: price breaks above R3 + price > 1d EMA34 (uptrend) + volume spike
        if close[i] > camarilla_r3_aligned[i] and close[i] > ema_34_1d_aligned[i] and volume_spike[i]:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: price breaks below S3 + price < 1d EMA34 (downtrend) + volume spike
        elif close[i] < camarilla_s3_aligned[i] and close[i] < ema_34_1d_aligned[i] and volume_spike[i]:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit conditions: price returns to mean (between R3 and S3) or loss of volume confirmation
        elif position == 1 and (close[i] < camarilla_r3_aligned[i] or not volume_spike[i]):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close[i] > camarilla_s3_aligned[i] or not volume_spike[i]):
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

name = "6h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0