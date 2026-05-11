#!/usr/bin/env python3
"""
1h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_v1
Hypothesis: Combine daily trend (price above/below SMA200) with Camarilla pivot breakouts (R3/S3) on 4h,
filtered by volume spikes, and executed on 1h for timing. Daily trend filters direction, 4h Camarilla
levels provide structure, volume spike confirms breakout strength. Works in bull/bear via daily trend.
Target: 60-150 total trades over 4 years on 1h timeframe.
"""

name = "1h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_v1"
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
    
    # === Daily Trend Filter (SMA200) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    sma200_1d = pd.Series(close_1d).rolling(window=200, min_periods=200).mean().values
    sma200_1d_aligned = align_htf_to_ltf(prices, df_1d, sma200_1d)
    
    # === 4H Camarilla Pivots (R3/S3 levels) ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate pivots using previous 4h bar
    prev_close = np.roll(close_4h, 1)
    prev_high = np.roll(high_4h, 1)
    prev_low = np.roll(low_4h, 1)
    prev_close[0] = close_4h[0]  # first bar uses current close
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_ = prev_high - prev_low
    r3 = pivot + range_ * 1.1 / 2
    s3 = pivot - range_ * 1.1 / 2
    
    # Align pivots to 1h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_4h, r3)
    s3_aligned = align_htf_to_ltf(prices, df_4h, s3)
    
    # === Volume Spike Filter (4h volume > 1.5x 20-period average) ===
    vol_ma_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    vol_spike = volume > (vol_ma_4h_aligned * 1.5)
    
    # === Session Filter: 08-20 UTC ===
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(200, 20)  # max of SMA200 and volume MA warmup
    
    for i in range(start_idx, n):
        if not session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Skip if any required data is invalid
        if (np.isnan(sma200_1d_aligned[i]) or 
            np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price > SMA200 (bullish trend) AND break above R3 AND volume spike
            if close[i] > sma200_1d_aligned[i] and high[i] > r3_aligned[i] and vol_spike[i]:
                signals[i] = 0.20
                position = 1
            # Short: price < SMA200 (bearish trend) AND break below S3 AND volume spike
            elif close[i] < sma200_1d_aligned[i] and low[i] < s3_aligned[i] and vol_spike[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price crosses below SMA200 OR breaks below S3 (stop)
            if close[i] < sma200_1d_aligned[i] or low[i] < s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20  # maintain position
        elif position == -1:
            # Short exit: price crosses above SMA200 OR breaks above R3 (stop)
            if close[i] > sma200_1d_aligned[i] or high[i] > r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20  # maintain position
    
    return signals