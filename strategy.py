#!/usr/bin/env python3
"""
4H_Camarilla_R3_S3_Trend_20EMA_Volume
Hypothesis: Use Camarilla R3/S3 levels from 1d as breakout levels with 20-period EMA trend filter and volume confirmation.
Goes long when price breaks above R3 with uptrend and volume, short when breaks below S3 with downtrend and volume.
Exits when price returns to the Camarilla pivot level (central level) or reverses trend.
Designed for 4h timeframe to target 20-50 trades/year, avoiding overtrading while capturing meaningful moves.
Works in bull markets (breaks above R3 in uptrend) and bear markets (breaks below S3 in downtrend).
"""
name = "4H_Camarilla_R3_S3_Trend_20EMA_Volume"
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
    
    # Get 1D data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each day: based on previous day's OHLC
    # Camarilla formulas:
    # H4 = close + 1.5 * (high - low)
    # H3 = close + 1.0 * (high - low)
    # H2 = close + 0.75 * (high - low)
    # H1 = close + 0.5 * (high - low)
    # L1 = close - 0.5 * (high - low)
    # L2 = close - 0.75 * (high - low)
    # L3 = close - 1.0 * (high - low)
    # L4 = close - 1.5 * (high - low)
    # Pivot = (high + low + close) / 3
    
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate levels
    R3 = prev_close + 1.0 * (prev_high - prev_low)
    S3 = prev_close - 1.0 * (prev_high - prev_low)
    Pivot = (prev_high + prev_low + prev_close) / 3.0
    
    # Align to 4h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    Pivot_aligned = align_htf_to_ltf(prices, df_1d, Pivot)
    
    # 20-period EMA for trend filter on 4h close
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume filter: current volume > 1.5 x 20-period average volume
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # EMA20 and volume filter need 20 periods
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(Pivot_aligned[i]) or np.isnan(ema_20[i]) or 
            np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3, uptrend (price > EMA20), and volume confirmation
            if (close[i] > R3_aligned[i] and 
                close[i] > ema_20[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3, downtrend (price < EMA20), and volume confirmation
            elif (close[i] < S3_aligned[i] and 
                  close[i] < ema_20[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to pivot level or trend reverses
            if close[i] <= Pivot_aligned[i] or close[i] < ema_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to pivot level or trend reverses
            if close[i] >= Pivot_aligned[i] or close[i] > ema_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals