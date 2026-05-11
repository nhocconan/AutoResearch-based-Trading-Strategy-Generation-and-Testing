#!/usr/bin/env python3
"""
1h_Camarilla_Pivot_Breakout_4hTrend_v1
Hypothesis: Uses Camarilla pivot breakouts on 1h with 4h trend filter and volume confirmation.
Long when price breaks above R3 in uptrend (4h EMA50 rising), short when breaks below S3 in downtrend.
Reduces whipsaw by requiring alignment with 4h trend. Targets 15-30 trades/year via tight entry conditions.
Works in bull/bear markets by following the 4h trend direction only.
"""

name = "1h_Camarilla_Pivot_Breakout_4hTrend_v1"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 1h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h EMA50 for trend
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Previous day's high, low, close for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels using previous day's OHLC
    high_prev = df_1d['high'].values
    low_prev = df_1d['low'].values
    close_prev = df_1d['close'].values
    
    # Camarilla multipliers
    R3 = close_prev + (high_prev - low_prev) * 1.1 / 2
    S3 = close_prev - (high_prev - low_prev) * 1.1 / 2
    
    # Align Camarilla levels to 1h
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # Volume filter: 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(volume_filter[i])):
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter: 4h EMA50 slope (using 3-bar change)
        if i >= 3:
            ema_slope = ema_50_4h_aligned[i] - ema_50_4h_aligned[i-3]
            uptrend = ema_slope > 0
            downtrend = ema_slope < 0
        else:
            uptrend = False
            downtrend = False
        
        if position == 0:
            # Look for breakouts with volume confirmation
            if uptrend and volume_filter[i]:
                # Long breakout above R3
                if close[i] > R3_aligned[i]:
                    signals[i] = 0.20
                    position = 1
            elif downtrend and volume_filter[i]:
                # Short breakdown below S3
                if close[i] < S3_aligned[i]:
                    signals[i] = -0.20
                    position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: price crosses below EMA50 or reverses below R3
                if close[i] < ema_50_4h_aligned[i] or close[i] < R3_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            elif position == -1:
                # Exit short: price crosses above EMA50 or reverses above S3
                if close[i] > ema_50_4h_aligned[i] or close[i] > S3_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals