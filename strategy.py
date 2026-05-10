#!/usr/bin/env python3
# 4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike
# Hypothesis: Using daily trend (EMA34) and volume spike (2x average) filters with Camarilla R3/S3 breakouts on 4h
# provides high-probability entries aligned with higher timeframe momentum, reducing false breakouts.
# Designed to work in both bull and bear markets by following daily trend direction.
# Target: 20-40 trades/year to minimize fee drag on 4h timeframe.

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike"
timeframe = "4h"
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
    
    # Daily trend filter (EMA34)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_1d_up = close_1d > ema34_1d
    trend_1d_down = close_1d < ema34_1d
    
    # Align daily trend to 4h
    trend_1d_up_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_up.astype(float))
    trend_1d_down_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_down.astype(float))
    
    # Volume spike filter (2x 24-period average)
    vol_ma = np.full(n, np.nan)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 24:
            vol_sum -= volume[i-24]
        if i >= 23:
            vol_ma[i] = vol_sum / 24
    volume_spike = volume > (2.0 * vol_ma)
    
    # Calculate Camarilla levels from previous daily bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Shift to get previous daily bar values
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Calculate Camarilla levels (R3, S3)
    R3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    S3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(trend_1d_up_aligned[i]) or np.isnan(trend_1d_down_aligned[i]) or
            np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3 with volume spike and daily uptrend
            if (high[i] > R3_aligned[i] and
                trend_1d_up_aligned[i] > 0.5 and
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 with volume spike and daily downtrend
            elif (low[i] < S3_aligned[i] and
                  trend_1d_down_aligned[i] > 0.5 and
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price breaks below S3 or daily trend turns down
            if (low[i] < S3_aligned[i] or
                trend_1d_up_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price breaks above R3 or daily trend turns up
            if (high[i] > R3_aligned[i] or
                trend_1d_down_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals