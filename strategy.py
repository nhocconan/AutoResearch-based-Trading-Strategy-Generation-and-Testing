#!/usr/bin/env python3
name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike"
timeframe = "4h"
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
    
    # Get 1d data for trend filter (daily EMA34)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Daily EMA34 for trend
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_up_1d = close_1d > ema34_1d
    
    # Get 1d data for Camarilla levels (R1, S1)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_cam = df_1d['close'].values
    
    # Calculate Camarilla levels (R1, S1) from previous day
    R1 = np.full(len(high_1d), np.nan)
    S1 = np.full(len(high_1d), np.nan)
    
    for i in range(1, len(high_1d)):
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d_cam[i-1]
        range_val = prev_high - prev_low
        if range_val > 0:
            R1[i] = prev_close + range_val * 1.1 / 12
            S1[i] = prev_close - range_val * 1.1 / 12
    
    # Get 1d data for volume average (20-day)
    vol_1d = df_1d['volume'].values
    vol_ma20_1d = np.full(len(vol_1d), np.nan)
    for i in range(len(vol_1d)):
        if i < 20:
            if i > 0:
                vol_ma20_1d[i] = np.mean(vol_1d[:i+1])
        else:
            vol_ma20_1d[i] = np.mean(vol_1d[i-19:i+1])
    
    # Align indicators to 4h timeframe
    trend_up_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_up_1d)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    vol_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20_1d)
    
    # Volume moving average (20-period) for 4h confirmation
    vol_ma20_4h = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            if i > 0:
                vol_ma20_4h[i] = np.mean(volume[:i+1])
        else:
            vol_ma20_4h[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 34)  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(R1_aligned[i]) or 
            np.isnan(S1_aligned[i]) or
            np.isnan(trend_up_1d_aligned[i]) or
            np.isnan(vol_ma20_1d_aligned[i]) or
            np.isnan(vol_ma20_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 + daily uptrend + volume confirmation (both 1d and 4h)
            if (close[i] > R1_aligned[i] and 
                trend_up_1d_aligned[i] and 
                volume[i] > 1.5 * vol_ma20_4h[i] and
                vol_1d[-1] > vol_ma20_1d_aligned[i] if len(vol_1d) > 0 else False):  # Daily volume confirmation
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + daily downtrend + volume confirmation
            elif (close[i] < S1_aligned[i] and 
                  not trend_up_1d_aligned[i] and 
                  volume[i] > 1.5 * vol_ma20_4h[i] and
                  vol_1d[-1] < vol_ma20_1d_aligned[i] if len(vol_1d) > 0 else False):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below S1 or trend changes
            if (close[i] < S1_aligned[i] or 
                not trend_up_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above R1 or trend changes
            if (close[i] > R1_aligned[i] or 
                trend_up_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals