#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_1d_camarilla_breakout_volume"
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
    
    # Weekly trend filter: EMA(21) - long-term trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    ema_21_1w = pd.Series(df_1w['close']).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Daily volatility for CAMARILLA calculation (using previous day)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    prev_close_1d = df_1d['close'].shift(1).values
    prev_high_1d = df_1d['high'].shift(1).values
    prev_low_1d = df_1d['low'].shift(1).values
    H_minus_L_1d = prev_high_1d - prev_low_1d
    
    # CAMARILLA LEVELS (daily)
    R3 = prev_close_1d + H_minus_L_1d * 1.1 / 4  # Strong resistance
    S3 = prev_close_1d - H_minus_L_1d * 1.1 / 4  # Strong support
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # Volume filter: current 6h volume > 48h average volume (2 days)
    vol_2d_avg = pd.Series(df_1d['volume']).rolling(window=2, min_periods=2).mean().values
    vol_2d_aligned = align_htf_to_ltf(prices, df_1d, vol_2d_avg)
    volume_filter = volume > vol_2d_aligned
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if not ready
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(ema_21_1w_aligned[i]) or np.isnan(volume_filter[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Apply session filter
        if not session_filter[i]:
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Long: price breaks above daily R3 with volume and above weekly EMA21
        long_signal = (close[i] > R3_aligned[i] and volume_filter[i] and close[i] > ema_21_1w_aligned[i])
        
        # Short: price breaks below daily S3 with volume and below weekly EMA21
        short_signal = (close[i] < S3_aligned[i] and volume_filter[i] and close[i] < ema_21_1w_aligned[i])
        
        # Exit: price returns to midpoint between daily R4/S4
        R4 = prev_close_1d + H_minus_L_1d * 1.1 / 2
        S4 = prev_close_1d - H_minus_L_1d * 1.1 / 2
        R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
        S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
        midpoint = (R4_aligned + S4_aligned) / 2
        
        exit_long = (position == 1 and close[i] < midpoint[i])
        exit_short = (position == -1 and close[i] > midpoint[i])
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals