# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_camarilla_breakout_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Daily trend: EMA(21)
    ema_21 = pd.Series(df_1d['close']).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_aligned = align_htf_to_ltf(prices, df_1d, ema_21)
    
    # Previous daily bar's OHLC for Camarilla calculation
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    H_minus_L = prev_high - prev_low
    # Camarilla levels: R3 (strong resistance), S3 (strong support)
    R3 = prev_close + H_minus_L * 1.1 / 4
    S3 = prev_close - H_minus_L * 1.1 / 4
    
    # Map daily levels to 12h bars
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # Volume confirmation: current 12h volume > 20-period average of daily volume
    vol_1d_aligned = align_htf_to_ltf(prices, df_1d, df_1d['volume'].values)
    vol_ma = pd.Series(vol_1d_aligned).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if not ready
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(ema_21_aligned[i]) or np.isnan(volume_filter[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Long: price breaks above R3 with volume and above daily EMA
        long_signal = (close[i] > R3_aligned[i] and volume_filter[i] and close[i] > ema_21_aligned[i])
        
        # Short: price breaks below S3 with volume and below daily EMA
        short_signal = (close[i] < S3_aligned[i] and volume_filter[i] and close[i] < ema_21_aligned[i])
        
        # Exit: price returns to midpoint between R2/S2
        H_minus_L_1d = (df_1d['high'].shift(1) - df_1d['low'].shift(1)).values
        R2 = df_1d['close'].shift(1).values + H_minus_L_1d * 1.1 / 6
        S2 = df_1d['close'].shift(1).values - H_minus_L_1d * 1.1 / 6
        R2_aligned = align_htf_to_ltf(prices, df_1d, R2)
        S2_aligned = align_htf_to_ltf(prices, df_1d, S2)
        midpoint = (R2_aligned + S2_aligned) / 2
        
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