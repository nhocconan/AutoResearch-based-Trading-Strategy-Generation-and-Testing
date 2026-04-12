#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_camarilla_breakout_volume_v2"
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
    
    # Get 12h data for trend and Camarilla levels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # 12h trend: EMA(21)
    ema_21 = pd.Series(df_12h['close']).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_aligned = align_htf_to_ltf(prices, df_12h, ema_21)
    
    # Previous 12h bar's OHLC for Camarilla calculation
    prev_close = df_12h['close'].shift(1).values
    prev_high = df_12h['high'].shift(1).values
    prev_low = df_12h['low'].shift(1).values
    
    H_minus_L = prev_high - prev_low
    # Camarilla levels: R3 (strong resistance), S3 (strong support)
    R3 = prev_close + H_minus_L * 1.1 / 4
    S3 = prev_close - H_minus_L * 1.1 / 4
    
    # Map 12h levels to 4h bars
    R3_aligned = align_htf_to_ltf(prices, df_12h, R3)
    S3_aligned = align_htf_to_ltf(prices, df_12h, S3)
    
    # Volume confirmation: current 4h volume > 20-period average of 12h volume
    vol_12h_aligned = align_htf_to_ltf(prices, df_12h, df_12h['volume'].values)
    vol_ma = pd.Series(vol_12h_aligned).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if not ready
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(ema_21_aligned[i]) or np.isnan(volume_filter[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Long: price breaks above R3 with volume and above 12h EMA
        long_signal = (close[i] > R3_aligned[i] and volume_filter[i] and close[i] > ema_21_aligned[i])
        
        # Short: price breaks below S3 with volume and below 12h EMA
        short_signal = (close[i] < S3_aligned[i] and volume_filter[i] and close[i] < ema_21_aligned[i])
        
        # Exit: price returns to midpoint between R2/S2
        H_minus_L_12h = (df_12h['high'].shift(1) - df_12h['low'].shift(1)).values
        R2 = df_12h['close'].shift(1).values + H_minus_L_12h * 1.1 / 6
        S2 = df_12h['close'].shift(1).values - H_minus_L_12h * 1.1 / 6
        R2_aligned = align_htf_to_ltf(prices, df_12h, R2)
        S2_aligned = align_htf_to_ltf(prices, df_12h, S2)
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