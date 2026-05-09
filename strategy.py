#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for daily trend and volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Daily EMA34 for trend filter (requires completion of daily candle)
    daily_ema34 = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    daily_ema34_4h = align_htf_to_ltf(prices, df_1d, daily_ema34)
    
    # Daily volume average for volume filter
    daily_vol_avg = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    daily_vol_avg_4h = align_htf_to_ltf(prices, df_1d, daily_vol_avg)
    
    # Get daily data for Camarilla pivot calculation (from previous day)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_range = prev_high - prev_low
    
    # Camarilla levels
    R3 = prev_close + 1.1 * prev_range / 6
    S3 = prev_close - 1.1 * prev_range / 6
    R4 = prev_close + 1.1 * prev_range / 2
    S4 = prev_close - 1.1 * prev_range / 2
    
    # Align Camarilla levels to 4h timeframe
    R3_4h = align_htf_to_ltf(prices, df_1d, R3)
    S3_4h = align_htf_to_ltf(prices, df_1d, S3)
    R4_4h = align_htf_to_ltf(prices, df_1d, R4)
    S4_4h = align_htf_to_ltf(prices, df_1d, S4)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(R3_4h[i]) or np.isnan(S3_4h[i]) or 
            np.isnan(R4_4h[i]) or np.isnan(S4_4h[i]) or 
            np.isnan(daily_ema34_4h[i]) or np.isnan(daily_vol_avg_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: current 4h volume > 1.5x daily average volume
        vol_ok = volume[i] > 1.5 * daily_vol_avg_4h[i]
        
        # Session filter: 08-20 UTC (reduce noise trades)
        hour = pd.Timestamp(prices['open_time'].iloc[i]).hour
        in_session = 8 <= hour <= 20
        
        if position == 0:
            # Long breakout: price breaks above R4 with daily uptrend
            if (close[i] > R4_4h[i] and 
                close[i] > daily_ema34_4h[i] and  # daily uptrend
                vol_ok and 
                in_session):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below S4 with daily downtrend
            elif (close[i] < S4_4h[i] and 
                  close[i] < daily_ema34_4h[i] and  # daily downtrend
                  vol_ok and 
                  in_session):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below R3 (mean reversion)
            if close[i] < R3_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above S3 (mean reversion)
            if close[i] > S3_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals