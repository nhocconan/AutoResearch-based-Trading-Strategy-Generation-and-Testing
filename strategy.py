#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend direction
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Camarilla levels from previous 1d OHLC
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_range = prev_high - prev_low
    
    # Camarilla levels R3/S3 and R4/S4
    R3 = prev_close + 1.1 * prev_range / 4
    S3 = prev_close - 1.1 * prev_range / 4
    R4 = prev_close + 1.1 * prev_range / 2
    S4 = prev_close - 1.1 * prev_range / 2
    
    # Align to 12h timeframe
    R3_12h = align_htf_to_ltf(prices, df_1d, R3)
    S3_12h = align_htf_to_ltf(prices, df_1d, S3)
    R4_12h = align_htf_to_ltf(prices, df_1d, R4)
    S4_12h = align_htf_to_ltf(prices, df_1d, S4)
    
    # Volume filter: above 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(R3_12h[i]) or np.isnan(S3_12h[i]) or 
            np.isnan(R4_12h[i]) or np.isnan(S4_12h[i]) or 
            np.isnan(ema_50_1d[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 1.5 * vol_ma[i]  # Volume confirmation
        
        # Session filter: 08-20 UTC (reduce noise trades)
        hour = pd.DatetimeIndex(prices['open_time']).hour[i]
        in_session = 8 <= hour <= 20
        
        if position == 0:
            # Long breakout: price breaks above R3 with 1d uptrend
            if (close[i] > R3_12h[i] and 
                close[i] > ema_50_1d[i] and  # 1d uptrend
                vol_ok and 
                in_session):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below S3 with 1d downtrend
            elif (close[i] < S3_12h[i] and 
                  close[i] < ema_50_1d[i] and  # 1d downtrend
                  vol_ok and 
                  in_session):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below S3 (mean reversion)
            if close[i] < S3_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above R3 (mean reversion)
            if close[i] > R3_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals