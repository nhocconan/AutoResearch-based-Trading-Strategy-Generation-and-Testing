#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_Camarilla_R3S3_Breakout_1wTrend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter and Camarilla calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # 1w EMA20 for trend direction (weekly trend)
    ema_20_1w = pd.Series(df_1w['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1d = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate Camarilla levels from previous weekly OHLC
    prev_close = df_1w['close'].shift(1).values
    prev_high = df_1w['high'].shift(1).values
    prev_low = df_1w['low'].shift(1).values
    prev_range = prev_high - prev_low
    
    # Camarilla levels R3/S3 (most relevant for breakouts)
    R3 = prev_close + 1.1 * prev_range / 4
    S3 = prev_close - 1.1 * prev_range / 4
    
    # Align to 1d timeframe
    R3_1d = align_htf_to_ltf(prices, df_1w, R3)
    S3_1d = align_htf_to_ltf(prices, df_1w, S3)
    
    # Volume filter: above 1.8x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(R3_1d[i]) or np.isnan(S3_1d[i]) or 
            np.isnan(ema_20_1w[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 1.8 * vol_ma[i]  # Volume confirmation
        
        if position == 0:
            # Long breakout: price breaks above R3 with 1w uptrend
            if (close[i] > R3_1d[i] and 
                close[i] > ema_20_1w[i] and  # 1w uptrend
                vol_ok):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below S3 with 1w downtrend
            elif (close[i] < S3_1d[i] and 
                  close[i] < ema_20_1w[i] and  # 1w downtrend
                  vol_ok):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below S3 (mean reversion)
            if close[i] < S3_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above R3 (mean reversion)
            if close[i] > R3_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals