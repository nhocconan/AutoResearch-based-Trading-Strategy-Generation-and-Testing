#!/usr/bin/env python3
name = "6h_12h_1d_Adaptive_Camarilla_Regime"
timeframe = "6h"
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
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # 12h EMA34 for trend
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_up_12h = close_12h > ema34_12h
    
    # Get 1d data for daily range and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily ATR(14) for volatility regime
    tr1 = np.zeros(len(df_1d))
    tr1[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(df_1d)):
        tr = high_1d[i] - low_1d[i]
        tr2 = abs(high_1d[i] - close_1d[i-1])
        tr3 = abs(low_1d[i] - close_1d[i-1])
        tr1[i] = max(tr, tr2, tr3)
    
    atr14_1d = pd.Series(tr1).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Daily range for adaptive position sizing
    daily_range = high_1d - low_1d
    range_ma20 = pd.Series(daily_range).rolling(window=20, min_periods=20).mean().values
    
    # Get 12h data for Camarilla levels (R3, S3)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h_cam = df_12h['close'].values
    
    # Calculate Camarilla levels (R3, S3) from previous 12h period
    R3 = np.full(len(high_12h), np.nan)
    S3 = np.full(len(high_12h), np.nan)
    
    for i in range(1, len(high_12h)):
        prev_high = high_12h[i-1]
        prev_low = low_12h[i-1]
        prev_close = close_12h_cam[i-1]
        range_val = prev_high - prev_low
        if range_val > 0:
            R3[i] = prev_close + range_val * 1.1 / 4
            S3[i] = prev_close - range_val * 1.1 / 4
    
    # Align indicators to 6h timeframe
    trend_up_12h_aligned = align_htf_to_ltf(prices, df_12h, trend_up_12h)
    atr14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr14_1d)
    range_ma20_aligned = align_htf_to_ltf(prices, df_1d, range_ma20)
    R3_aligned = align_htf_to_ltf(prices, df_12h, R3)
    S3_aligned = align_htf_to_ltf(prices, df_12h, S3)
    
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
            np.isnan(trend_up_12h_aligned[i]) or
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
            # Long: price breaks above R3 + uptrend + volume confirmation
            if (close[i] > R3_aligned[i] and 
                trend_up_12h_aligned[i] and 
                volume[i] > 1.3 * vol_ma20[i]):
                signals[i] = base_size
                position = 1
            # Short: price breaks below S3 + downtrend + volume confirmation
            elif (close[i] < S3_aligned[i] and 
                  not trend_up_12h_aligned[i] and 
                  volume[i] > 1.3 * vol_ma20[i]):
                signals[i] = -base_size
                position = -1
        elif position == 1:
            # Long exit: price breaks below S3 or trend changes or volatility spike
            if (close[i] < S3_aligned[i] or 
                not trend_up_12h_aligned[i] or
                atr14_1d_aligned[i] > 2.0 * range_ma20_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = base_size
        elif position == -1:
            # Short exit: price breaks above R3 or trend changes or volatility spike
            if (close[i] > R3_aligned[i] or 
                trend_up_12h_aligned[i] or
                atr14_1d_aligned[i] > 2.0 * range_ma20_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -base_size
    
    return signals