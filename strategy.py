#!/usr/bin/env python3
name = "12h_Camarilla_R3S3_Breakout_WeeklyTrend_Volume"
timeframe = "12h"
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Weekly EMA50 for trend
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_up_1w = close_1w > ema50_1w
    
    # Get daily data for weekly range and volatility
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Weekly ATR(14) for volatility regime (using daily data)
    tr1 = np.zeros(len(df_1d))
    tr1[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(df_1d)):
        tr = high_1d[i] - low_1d[i]
        tr2 = abs(high_1d[i] - close_1d[i-1])
        tr3 = abs(low_1d[i] - close_1d[i-1])
        tr1[i] = max(tr, tr2, tr3)
    
    atr14_1d = pd.Series(tr1).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Weekly range for adaptive position sizing
    weekly_range = high_1d - low_1d
    range_ma20 = pd.Series(weekly_range).rolling(window=20, min_periods=20).mean().values
    
    # Get daily data for Camarilla levels (R3, S3)
    high_1d_cam = df_1d['high'].values
    low_1d_cam = df_1d['low'].values
    close_1d_cam = df_1d['close'].values
    
    # Calculate Camarilla levels (R3, S3) from previous day
    R3 = np.full(len(high_1d_cam), np.nan)
    S3 = np.full(len(high_1d_cam), np.nan)
    
    for i in range(1, len(high_1d_cam)):
        prev_high = high_1d_cam[i-1]
        prev_low = low_1d_cam[i-1]
        prev_close = close_1d_cam[i-1]
        range_val = prev_high - prev_low
        if range_val > 0:
            R3[i] = prev_close + range_val * 1.1 / 4
            S3[i] = prev_close - range_val * 1.1 / 4
    
    # Align indicators to 12h timeframe
    trend_up_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_up_1w)
    atr14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr14_1d)
    range_ma20_aligned = align_htf_to_ltf(prices, df_1d, range_ma20)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
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
    
    start_idx = max(30, 50)  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(R3_aligned[i]) or 
            np.isnan(S3_aligned[i]) or
            np.isnan(trend_up_1w_aligned[i]) or
            np.isnan(atr14_1d_aligned[i]) or
            np.isnan(range_ma20_aligned[i]) or
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Adaptive position sizing based on volatility regime
        # Low volatility: smaller position, High volatility: larger position
        vol_ratio = atr14_1d_aligned[i] / range_ma20_aligned[i] if range_ma20_aligned[i] > 0 else 1.0
        # Scale position size: 0.15 in low vol, 0.30 in high vol
        base_size = 0.15 + 0.15 * min(vol_ratio, 1.0)  # Cap at 0.30
        
        if position == 0:
            # Long: price breaks above R3 + weekly uptrend + volume confirmation
            if (close[i] > R3_aligned[i] and 
                trend_up_1w_aligned[i] and 
                volume[i] > 1.3 * vol_ma20[i]):
                signals[i] = base_size
                position = 1
            # Short: price breaks below S3 + weekly downtrend + volume confirmation
            elif (close[i] < S3_aligned[i] and 
                  not trend_up_1w_aligned[i] and 
                  volume[i] > 1.3 * vol_ma20[i]):
                signals[i] = -base_size
                position = -1
        elif position == 1:
            # Long exit: price breaks below S3 or trend changes or volatility spike
            if (close[i] < S3_aligned[i] or 
                not trend_up_1w_aligned[i] or
                atr14_1d_aligned[i] > 2.0 * range_ma20_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = base_size
        elif position == -1:
            # Short exit: price breaks above R3 or trend changes or volatility spike
            if (close[i] > R3_aligned[i] or 
                trend_up_1w_aligned[i] or
                atr14_1d_aligned[i] > 2.0 * range_ma20_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -base_size
    
    return signals