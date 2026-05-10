#!/usr/bin/env python3
# 6h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike
# Hypothesis: Use daily Camarilla R3/S3 levels for breakout confirmation with 1d trend filter and volume spike.
# Long when price breaks above R3 with 1d uptrend and volume > 2x average; short when breaks below S3 with 1d downtrend and volume spike.
# Designed for low trade frequency (15-25/year) to avoid fee drag, works in bull/bear via trend filter.

name = "6h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

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
    
    # 1d data for Camarilla levels, trend, and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    open_1d = df_1d['open'].values
    
    # Calculate pivot point (PP)
    pp = (high_1d + low_1d + close_1d) / 3.0
    # Calculate daily range
    range_1d = high_1d - low_1d
    
    # Camarilla levels: R3/S3
    r3 = pp + range_1d * 1.1 / 2.0
    s3 = pp - range_1d * 1.1 / 2.0
    
    # 1d EMA34 trend
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_1d_up = close_1d > ema34_1d
    trend_1d_down = close_1d < ema34_1d
    
    # 1d volume spike: current volume > 2 * 20-period average
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (2.0 * vol_ma_1d)
    
    # Align 1d indicators to 6h
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    trend_1d_up_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_up.astype(float))
    trend_1d_down_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_down.astype(float))
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure enough history for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(trend_1d_up_aligned[i]) or np.isnan(trend_1d_down_aligned[i]) or
            np.isnan(volume_spike_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3 with 1d uptrend and volume spike
            if (close[i] > r3_aligned[i] and
                trend_1d_up_aligned[i] > 0.5 and
                volume_spike_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 with 1d downtrend and volume spike
            elif (close[i] < s3_aligned[i] and
                  trend_1d_down_aligned[i] > 0.5 and
                  volume_spike_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price re-enters below R3 or trend fails
            if (close[i] < r3_aligned[i] or
                trend_1d_up_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price re-enters above S3 or trend fails
            if (close[i] > s3_aligned[i] or
                trend_1d_down_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals