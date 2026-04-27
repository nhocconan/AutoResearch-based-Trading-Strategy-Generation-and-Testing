#!/usr/bin/env python3
"""
6h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike
Hypothesis: Camarilla R3/S3 breakouts with volume spike and daily trend filter capture
institutional breakout moves with controlled frequency. Uses R3/S3 for stronger breakouts
than R1/S1, reducing false signals. Works in bull/bear by filtering with 1d EMA34 trend.
Targets 15-30 trades/year on 6h to minimize fee drag.
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
    
    # Get 1d data for trend filter and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Camarilla levels from previous day
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    close_prev = df_1d['close'].shift(1).values
    
    # Camarilla R3, S3 levels (stronger breakout levels)
    R3 = close_prev + (high_prev - low_prev) * 1.1 * 6 / 12  # = close_prev + (high-low)*0.55
    S3 = close_prev - (high_prev - low_prev) * 1.1 * 6 / 12  # = close_prev - (high-low)*0.55
    
    # Align Camarilla levels to 6h timeframe (available after daily close)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # Volume spike: volume > 2.5 * 20-period average (strong confirmation)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for EMA and volume MA
    start_idx = max(35, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(ema34_1d_aligned[i]) or np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]):
            signals[i] = 0.0
            continue
        
        ema_trend = ema34_1d_aligned[i]
        r3_level = R3_aligned[i]
        s3_level = S3_aligned[i]
        vol_spike_val = vol_spike[i]
        
        if position == 0:
            # Long: break above R3 with volume spike and uptrend
            if close[i] > r3_level and vol_spike_val and close[i] > ema_trend:
                signals[i] = size
                position = 1
            # Short: break below S3 with volume spike and downtrend
            elif close[i] < s3_level and vol_spike_val and close[i] < ema_trend:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: close below S3 or trend turns down
            if close[i] < s3_level or close[i] < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: close above R3 or trend turns up
            if close[i] > r3_level or close[i] > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0