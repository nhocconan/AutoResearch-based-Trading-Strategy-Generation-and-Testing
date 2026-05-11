#!/usr/bin/env python3
name = "4h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike"
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
    
    # Get 1d data for trend filter and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d EMA34 for trend
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_up_1d = close_1d > ema34_1d
    
    # 1d ATR(14) for volatility filter
    tr = np.zeros(len(df_1d))
    tr[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(df_1d)):
        tr[i] = max(high_1d[i] - low_1d[i],
                    abs(high_1d[i] - close_1d[i-1]),
                    abs(low_1d[i] - close_1d[i-1]))
    atr14_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Get 4h data for Camarilla levels (R3, S3)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla levels (R3, S3) from previous 4h period
    R3 = np.full(len(high_4h), np.nan)
    S3 = np.full(len(high_4h), np.nan)
    for i in range(1, len(high_4h)):
        prev_high = high_4h[i-1]
        prev_low = low_4h[i-1]
        prev_close = close_4h[i-1]
        range_val = prev_high - prev_low
        if range_val > 0:
            R3[i] = prev_close + range_val * 1.1 / 4
            S3[i] = prev_close - range_val * 1.1 / 4
    
    # Align indicators to 4h timeframe
    trend_up_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_up_1d)
    atr14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr14_1d)
    R3_aligned = align_htf_to_ltf(prices, df_4h, R3)
    S3_aligned = align_htf_to_ltf(prices, df_4h, S3)
    
    # Volume moving average (20-period) for confirmation
    vol_ma20 = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            if i > 0:
                vol_ma20[i] = np.mean(volume[:i+1])
        else:
            vol_ma20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 34)  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(R3_aligned[i]) or 
            np.isnan(S3_aligned[i]) or
            np.isnan(trend_up_1d_aligned[i]) or
            np.isnan(atr14_1d_aligned[i]) or
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        volume_confirm = volume[i] > 1.5 * vol_ma20[i]
        
        if position == 0:
            # Long: price breaks above R3 + uptrend + volume confirmation
            if (close[i] > R3_aligned[i] and 
                trend_up_1d_aligned[i] and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 + downtrend + volume confirmation
            elif (close[i] < S3_aligned[i] and 
                  not trend_up_1d_aligned[i] and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below S3 or trend changes
            if (close[i] < S3_aligned[i] or 
                not trend_up_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above R3 or trend changes
            if (close[i] > R3_aligned[i] or 
                trend_up_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals