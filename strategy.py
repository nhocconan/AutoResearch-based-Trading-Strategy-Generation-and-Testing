#!/usr/bin/env python3
name = "6h_Camarilla_R3_S3_Breakout_1wTrend_Volume"
timeframe = "6h"
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
    
    # 1w data for weekly trend filter (10-period EMA)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_10_1w = pd.Series(close_1w).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # 1d data for Camarilla pivot calculation (R3 and S3 levels)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Camarilla R3 and S3 using previous day's OHLC (avoid look-ahead)
    # R3 = C + (H-L) * 1.1/2
    # S3 = C - (H-L) * 1.1/2
    prev_close = np.concatenate([[np.nan], close_1d[:-1]])
    prev_high = np.concatenate([[np.nan], high_1d[:-1]])
    prev_low = np.concatenate([[np.nan], low_1d[:-1]])
    
    camarilla_r3 = prev_close + ((prev_high - prev_low) * 1.1 / 2)
    camarilla_s3 = prev_close - ((prev_high - prev_low) * 1.1 / 2)
    
    # Volume spike detection (24-period average for 6h, ~4 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_spike = volume > (vol_ma * 2.0)  # Require strong volume spike
    
    # Align all indicators to 6h timeframe
    ema_10_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_10_1w)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(24, 10)  # Ensure enough data for volume MA and weekly EMA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_10_1w_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price breaks above R3 with volume spike and above weekly EMA10 (bullish trend)
            if (close[i] > camarilla_r3_aligned[i] and 
                vol_spike[i] and 
                close[i] > ema_10_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S3 with volume spike and below weekly EMA10 (bearish trend)
            elif (close[i] < camarilla_s3_aligned[i] and 
                  vol_spike[i] and 
                  close[i] < ema_10_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price closes below R3 or below weekly EMA10
            if close[i] < camarilla_r3_aligned[i] or close[i] < ema_10_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price closes above S3 or above weekly EMA10
            if close[i] > camarilla_s3_aligned[i] or close[i] > ema_10_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals