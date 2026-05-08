#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    close_prev = df_1d['close'].shift(1).values
    
    pivot = (high_prev + low_prev + close_prev) / 3
    range_prev = high_prev - low_prev
    
    # Camarilla levels
    R3 = pivot + (range_prev * 1.1 / 2)
    S3 = pivot - (range_prev * 1.1 / 2)
    R4 = pivot + (range_prev * 1.1)
    S4 = pivot - (range_prev * 1.1)
    
    # Daily trend: EMA34 > EMA89 for uptrend
    ema_34 = pd.Series(close_prev).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_89 = pd.Series(close_prev).ewm(span=89, adjust=False, min_periods=89).mean().values
    daily_uptrend = ema_34 > ema_89
    daily_downtrend = ema_34 < ema_89
    
    # Align all to 6h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    daily_uptrend_aligned = align_htf_to_ltf(prices, df_1d, daily_uptrend)
    daily_downtrend_aligned = align_htf_to_ltf(prices, df_1d, daily_downtrend)
    
    # Volume confirmation: current volume > 1.5x 24-period average (4 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1.0)
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(R4_aligned[i]) or np.isnan(S4_aligned[i]) or
            np.isnan(daily_uptrend_aligned[i]) or np.isnan(daily_downtrend_aligned[i]) or
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above R4 with daily uptrend and volume spike
            if (close[i] > R4_aligned[i] and 
                daily_uptrend_aligned[i] and
                vol_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below S4 with daily downtrend and volume spike
            elif (close[i] < S4_aligned[i] and 
                  daily_downtrend_aligned[i] and
                  vol_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below R3 or volume drops
            if (close[i] < R3_aligned[i] or vol_ratio[i] < 0.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above S3 or volume drops
            if (close[i] > S3_aligned[i] or vol_ratio[i] < 0.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals