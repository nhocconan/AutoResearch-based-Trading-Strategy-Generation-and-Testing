#!/usr/bin/env python3
"""
6h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike_v1
Hypothesis: Combines Camarilla pivot breakouts from the 1d timeframe with 1d trend filter and volume spikes.
In bull markets, upward breakouts from R3 with trend alignment and volume surge continue higher.
In bear markets, downward breakdowns from S3 with trend alignment and volume surge continue lower.
Volume spikes filter out false breakouts. Target: 50-150 trades over 4 years (12-37/year) on 6h timeframe.
"""

name = "6h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

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
    
    # === 1D Data for Camarilla Pivots and Trend ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla levels: R3, R4, S3, S4
    # R4 = Close + 1.5 * (High - Low)
    # R3 = Close + 1.1 * (High - Low)
    # S3 = Close - 1.1 * (High - Low)
    # S4 = Close - 1.5 * (High - Low)
    rng = prev_high - prev_low
    r3 = prev_close + 1.1 * rng
    r4 = prev_close + 1.5 * rng
    s3 = prev_close - 1.1 * rng
    s4 = prev_close - 1.5 * rng
    
    # Align Camarilla levels to 6h
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    r4_6h = align_htf_to_ltf(prices, df_1d, r4)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    s4_6h = align_htf_to_ltf(prices, df_1d, s4)
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # === Volume Spike Filter ===
    # Volume MA(20) on 6h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 1.0)  # volume / avg volume
    vol_spike = vol_ratio > 2.0  # at least 2x average volume
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(100, 34)
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(r3_6h[i]) or np.isnan(r4_6h[i]) or 
            np.isnan(s3_6h[i]) or np.isnan(s4_6h[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above R3 with uptrend and volume spike
            if (close[i] > r3_6h[i] and 
                ema34_1d_aligned[i] < close[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below S3 with downtrend and volume spike
            elif (close[i] < s3_6h[i] and 
                  ema34_1d_aligned[i] > close[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price falls back below R3 or trend reverses
            if close[i] < r3_6h[i] or ema34_1d_aligned[i] > close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price rises back above S3 or trend reverses
            if close[i] > s3_6h[i] or ema34_1d_aligned[i] < close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals