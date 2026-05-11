#!/usr/bin/env python3
name = "1h_Camarilla_R1_S1_Breakout_4hTrend_1dVol"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Camarilla calculation (previous day)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 6:
        return np.zeros(n)
    
    # Calculate previous day's high, low, close from 4h data
    # Group 4h bars by day (6 bars per day)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate daily OHLC from 4h data
    n_4h = len(df_4h)
    days = n_4h // 6
    if days < 1:
        return np.zeros(n)
    
    # Previous day's OHLC (excluding current incomplete day)
    prev_day_idx = max(0, days - 1)
    start_idx = prev_day_idx * 6
    end_idx = start_idx + 6
    
    if end_idx > len(high_4h):
        return np.zeros(n)
    
    prev_high = np.max(high_4h[start_idx:end_idx])
    prev_low = np.min(low_4h[start_idx:end_idx])
    prev_close = close_4h[end_idx - 1]  # Last 4h bar of previous day
    
    # Camarilla levels
    range_val = prev_high - prev_low
    if range_val <= 0:
        R1 = S1 = prev_close
    else:
        R1 = prev_close + (range_val * 1.1 / 12)
        S1 = prev_close - (range_val * 1.1 / 12)
    
    # Get 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_up_1d = close_1d > ema50_1d
    trend_up_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_up_1d)
    
    # Get 1d data for volume filter (average volume)
    vol_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Volume confirmation: current volume > 1.5x 1-day average volume
    volume_filter = volume > (1.5 * vol_avg_1d_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 6)  # Need enough data for EMA and volume
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(R1) or np.isnan(S1) or
            np.isnan(trend_up_1d_aligned[i]) or
            np.isnan(vol_avg_1d_aligned[i]) or
            np.isnan(hours[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if position == 0:
            # Long: Camarilla R1 breakout + 1d uptrend + volume filter + session
            if (close[i] > R1 and 
                trend_up_1d_aligned[i] and 
                volume_filter[i] and 
                in_session):
                signals[i] = 0.20
                position = 1
            # Short: Camarilla S1 breakdown + 1d downtrend + volume filter + session
            elif (close[i] < S1 and 
                  not trend_up_1d_aligned[i] and 
                  volume_filter[i] and 
                  in_session):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: Camarilla S1 breakdown OR 1d trend turns down
            if (close[i] < S1 or not trend_up_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: Camarilla R1 breakout OR 1d trend turns up
            if (close[i] > R1 or trend_up_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals