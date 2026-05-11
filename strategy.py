#!/usr/bin/env python3
"""
1h_Camarilla_R3_S3_Breakout_4hTrend_VolumeSpike
Hypothesis: Use 4h trend (EMA50) and 4h volume spike for signal direction, with 1h entry timing at Camarilla R3/S3 levels. Works in both bull and bear markets by following the 4h trend direction. The 4h timeframe reduces noise, while 1h provides precise entry timing. Target: 15-35 trades per year on 1h timeframe.
"""

name = "1h_Camarilla_R3_S3_Breakout_4hTrend_VolumeSpike"
timeframe = "1h"
leverage = 1.0

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
    
    # === 4H Data for Trend Filter and Volume Spike ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Trend filter: EMA50 on 4h close
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Volume spike: current 4h volume > 2x 20-period average
    vol_ma_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    volume_spike_4h = volume_4h > (vol_ma_4h * 2.0)
    volume_spike_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_spike_4h)
    
    # === 1D Data for Camarilla Pivots ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = high_1d[0]  # First day uses same day's high
    prev_low[0] = low_1d[0]    # First day uses same day's low
    prev_close[0] = close_1d[0] # First day uses same day's close
    
    # Camarilla levels: R3/S3 = C ± (H-L) * 1.1/2
    rang = prev_high - prev_low
    r3 = prev_close + rang * 1.1 / 2
    s3 = prev_close - rang * 1.1 / 2
    
    # Align 1D Camarilla levels to 1h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or 
            np.isnan(ema_50_4h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R3 AND 4h uptrend (price > EMA50) AND 4h volume spike
            if close[i] > r3_aligned[i] and close[i] > ema_50_4h_aligned[i] and volume_spike_4h_aligned[i]:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S3 AND 4h downtrend (price < EMA50) AND 4h volume spike
            elif close[i] < s3_aligned[i] and close[i] < ema_50_4h_aligned[i] and volume_spike_4h_aligned[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price crosses below EMA50 OR reverses below R3
            if close[i] < ema_50_4h_aligned[i] or close[i] < r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20  # maintain position
        elif position == -1:
            # Short exit: price crosses above EMA50 OR reverses above S3
            if close[i] > ema_50_4h_aligned[i] or close[i] > s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20  # maintain position
    
    return signals