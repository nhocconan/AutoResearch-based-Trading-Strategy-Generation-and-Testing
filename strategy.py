# [EXPERIMENT #153999] 6h_Camarilla_R3_S3_Breakout_12hTrend_VolumeSpike_v2
# Hypothesis: Use 12h Camarilla R3/S3 breakouts with 12h EMA50 trend filter and volume confirmation on 6h timeframe.
# This targets 12-37 trades/year to avoid fee drag, using 12h timeframe for structure and 6h for execution.
# Should work in both bull and bear markets by only trading breakouts in the direction of 12h trend.

#!/usr/bin/env python3
name = "6h_Camarilla_R3_S3_Breakout_12hTrend_VolumeSpike_v2"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for Camarilla pivots and trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h Camarilla levels from previous 12h bar's OHLC
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Previous 12h bar's range
    range_12h = high_12h - low_12h
    
    # Calculate Camarilla R3 and S3 levels (most commonly used for breakouts)
    camarilla_r3 = close_12h + (range_12h * 1.1 / 2)
    camarilla_s3 = close_12h - (range_12h * 1.1 / 2)
    
    # Align Camarilla levels to 6h timeframe (using previous 12h bar's values)
    r3_6h = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    s3_6h = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    
    # 12h EMA50 for trend filter
    close_12h_series = pd.Series(close_12h)
    ema_12h = close_12h_series.ewm(span=50, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Volume filter: current volume > 2.0x 50-period average (higher threshold = fewer trades)
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_filter = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or 
            np.isnan(ema_12h_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R3 AND above 12h EMA50 (uptrend) AND volume spike
            if close[i] > r3_6h[i] and close[i] > ema_12h_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 AND below 12h EMA50 (downtrend) AND volume spike
            elif close[i] < s3_6h[i] and close[i] < ema_12h_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price falls below S3 OR below 12h EMA50 (trend change)
            if close[i] < s3_6h[i] or close[i] < ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price rises above R3 OR above 12h EMA50 (trend change)
            if close[i] > r3_6h[i] or close[i] > ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals