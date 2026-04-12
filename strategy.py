#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_camarilla_breakout_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Get daily data for Camarilla levels (calculated from previous day)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate previous day's Camarilla levels
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    H_minus_L = prev_high - prev_low
    R4 = prev_close + H_minus_L * 1.1 / 2
    R3 = prev_close + H_minus_L * 1.1 / 4
    S3 = prev_close - H_minus_L * 1.1 / 4
    S4 = prev_close - H_minus_L * 1.1 / 2
    
    # Align daily levels to 1h timeframe
    R4_1h = align_htf_to_ltf(open_time, df_1d, R4)
    R3_1h = align_htf_to_ltf(open_time, df_1d, R3)
    S3_1h = align_htf_to_ltf(open_time, df_1d, S3)
    S4_1h = align_htf_to_ltf(open_time, df_1d, S4)
    
    # Get 4h trend filter (EMA21)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 21:
        return np.zeros(n)
    ema_4h = pd.Series(df_4h['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_4h_1h = align_htf_to_ltf(open_time, df_4h, ema_4h)
    
    # Volume filter: current volume > 20-period EMA of volume
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ema
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(open_time).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if not ready
        if (np.isnan(R4_1h[i]) or np.isnan(R3_1h[i]) or 
            np.isnan(S3_1h[i]) or np.isnan(S4_1h[i]) or
            np.isnan(ema_4h_1h[i])):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Long: price breaks above R4 in uptrend (price > EMA) with volume during session
        long_signal = (close[i] > R4_1h[i] and 
                      close[i] > ema_4h_1h[i] and 
                      volume_filter[i] and 
                      session_filter[i])
        
        # Short: price breaks below S4 in downtrend (price < EMA) with volume during session
        short_signal = (close[i] < S4_1h[i] and 
                       close[i] < ema_4h_1h[i] and 
                       volume_filter[i] and 
                       session_filter[i])
        
        # Exit: price returns to mid-point (S3/R3)
        exit_long = (position == 1 and close[i] < (R3_1h[i] + S3_1h[i]) / 2)
        exit_short = (position == -1 and close[i] > (R3_1h[i] + S3_1h[i]) / 2)
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.20
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.20
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals