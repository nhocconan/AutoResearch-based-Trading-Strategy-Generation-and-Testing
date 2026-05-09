#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSqueeze"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot calculation
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
    R2 = prev_close + 1.1 * prev_range / 6
    S2 = prev_close - 1.1 * prev_range / 6
    
    # Align to 4h timeframe
    R1_4h = align_htf_to_ltf(prices, df_1d, R1)
    S1_4h = align_htf_to_ltf(prices, df_1d, S1)
    R2_4h = align_htf_to_ltf(prices, df_1d, R2)
    S2_4h = align_htf_to_ltf(prices, df_1d, S2)
    
    # Get daily trend filter (1d EMA34)
    daily_close = df_1d['close'].values
    daily_ema = pd.Series(daily_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    daily_ema_4h = align_htf_to_ltf(prices, df_1d, daily_ema)
    
    # Volume squeeze filter: volume < 0.5x 20-period average (low volume = consolidation)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(R1_4h[i]) or np.isnan(S1_4h[i]) or 
            np.isnan(R2_4h[i]) or np.isnan(S2_4h[i]) or 
            np.isnan(daily_ema_4h[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_squeeze = volume[i] < 0.5 * vol_ma[i]  # Volume squeeze = consolidation breakout
        
        # Session filter: 08-20 UTC (reduce noise trades)
        hour = pd.Timestamp(prices['open_time'].iloc[i]).hour
        in_session = 8 <= hour <= 20
        
        if position == 0:
            # Long breakout: price breaks above R2 with daily uptrend and volume squeeze
            if (close[i] > R2_4h[i] and 
                close[i] > daily_ema_4h[i] and  # daily uptrend
                vol_squeeze and 
                in_session):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below S2 with daily downtrend and volume squeeze
            elif (close[i] < S2_4h[i] and 
                  close[i] < daily_ema_4h[i] and  # daily downtrend
                  vol_squeeze and 
                  in_session):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below R1 (mean reversion to pivot)
            if close[i] < R1_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above S1 (mean reversion to pivot)
            if close[i] > S1_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals