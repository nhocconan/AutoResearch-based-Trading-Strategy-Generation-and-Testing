#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_Camarilla_R1S1_Breakout_4hTrend_VolumeSpike_v2"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 4h EMA50 for trend direction
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1h = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_range = prev_high - prev_low
    
    # Camarilla levels
    R1 = prev_close + 1.1 * prev_range / 12
    S1 = prev_close - 1.1 * prev_range / 12
    
    # Align to 1h timeframe
    R1_1h = align_htf_to_ltf(prices, df_1d, R1)
    S1_1h = align_htf_to_ltf(prices, df_1d, S1)
    
    # Volume filter: above 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(R1_1h[i]) or np.isnan(S1_1h[i]) or 
            np.isnan(ema_50_1h[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 1.5 * vol_ma[i]  # Volume confirmation
        
        # Session filter: 08-20 UTC (reduce noise trades)
        hour = pd.DatetimeIndex(prices['open_time']).hour[i]
        in_session = 8 <= hour <= 20
        
        if position == 0:
            # Long breakout: price breaks above R1 with 4h uptrend
            if (close[i] > R1_1h[i] and 
                close[i] > ema_50_1h[i] and  # 4h uptrend
                vol_ok and 
                in_session):
                signals[i] = 0.20
                position = 1
            # Short breakdown: price breaks below S1 with 4h downtrend
            elif (close[i] < S1_1h[i] and 
                  close[i] < ema_50_1h[i] and  # 4h downtrend
                  vol_ok and 
                  in_session):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below S1 (mean reversion)
            if close[i] < S1_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price rises back above R1 (mean reversion)
            if close[i] > R1_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals