#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_camarilla_breakout_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Camarilla levels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Previous 12h bar's OHLC for Camarilla calculation
    prev_close = df_12h['close'].shift(1).values
    prev_high = df_12h['high'].shift(1).values
    prev_low = df_12h['low'].shift(1).values
    
    H_minus_L = prev_high - prev_low
    # Camarilla levels: R4 (strong resistance), S4 (strong support)
    R4 = prev_close + H_minus_L * 1.1 / 2
    S4 = prev_close - H_minus_L * 1.1 / 2
    
    # Map 12h Camarilla levels to each 4h bar using proper alignment
    R4_aligned = align_htf_to_ltf(prices, df_12h, R4)
    S4_aligned = align_htf_to_ltf(prices, df_12h, S4)
    
    # Volume confirmation: current 4h volume > 20-period average of 12h volume
    vol_12h_aligned = align_htf_to_ltf(prices, df_12h, df_12h['volume'].values)
    vol_ma = pd.Series(vol_12h_aligned).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if not ready
        if (np.isnan(R4_aligned[i]) or np.isnan(S4_aligned[i]) or 
            np.isnan(volume_filter[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Long: price breaks above R4 (strong resistance) with volume
        long_signal = (close[i] > R4_aligned[i] and volume_filter[i])
        
        # Short: price breaks below S4 (strong support) with volume
        short_signal = (close[i] < S4_aligned[i] and volume_filter[i])
        
        # Exit: price returns to midpoint between R3/S3
        # Calculate R3/S3 for exit condition
        H_minus_L_12h = (df_12h['high'].shift(1) - df_12h['low'].shift(1)).values
        R3 = df_12h['close'].shift(1).values + H_minus_L_12h * 1.1 / 4
        S3 = df_12h['close'].shift(1).values - H_minus_L_12h * 1.1 / 4
        R3_aligned = align_htf_to_ltf(prices, df_12h, R3)
        S3_aligned = align_htf_to_ltf(prices, df_12h, S3)
        midpoint = (R3_aligned + S3_aligned) / 2
        
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