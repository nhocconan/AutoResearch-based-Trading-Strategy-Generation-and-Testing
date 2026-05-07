#!/usr/bin/env python3
"""
12H_Camarilla_R3_S3_Breakout_1DTrend_Volume
Hypothesis: Buy when price breaks above Camarilla R3 (resistance) on 12h chart with daily uptrend and volume confirmation.
Sell/short when price breaks below S3 (support) with daily downtrend and volume.
Camarilla levels provide institutional support/resistance, daily trend filters counter-trend trades,
volume avoids breakouts in low conviction. Designed for 12h timeframe to target 12-37 trades/year.
Works in bull markets (buys breakouts in uptrend) and bear markets (sells breakdowns in downtrend).
"""

name = "12H_Camarilla_R3_S3_Breakout_1DTrend_Volume"
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
    
    # Get 12H data for Camarilla calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 12h bar
    # Typical Price = (H + L + C) / 3
    typical_price = (df_12h['high'] + df_12h['low'] + df_12h['close']) / 3
    # Range = H - L
    range_12h = df_12h['high'] - df_12h['low']
    # Camarilla levels
    R3 = typical_price + range_12h * 1.1 / 4
    S3 = typical_price - range_12h * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe (already at 12h resolution)
    R3_12h = R3.values
    S3_12h = S3.values
    R3_aligned = align_htf_to_ltf(prices, df_12h, R3_12h)
    S3_aligned = align_htf_to_ltf(prices, df_12h, S3_12h)
    
    # Get 1D data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1D EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: current 12h volume > 1.5 x 20-period average volume
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3, daily uptrend, and volume confirmation
            if (close[i] > R3_aligned[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3, daily downtrend, and volume confirmation
            elif (close[i] < S3_aligned[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below S3 (support)
            if close[i] < S3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above R3 (resistance)
            if close[i] > R3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals