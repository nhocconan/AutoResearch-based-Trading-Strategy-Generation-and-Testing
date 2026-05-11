#!/usr/bin/env python3
name = "4h_Camarilla_R3S3_Breakout_1wTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels from weekly data
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly pivot calculation (close to close)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Camarilla R3, S3 levels
    pivot_weekly = (weekly_high + weekly_low + weekly_close) / 3.0
    r3_weekly = weekly_close + (weekly_high - weekly_low) * 1.1 / 4
    s3_weekly = weekly_close - (weekly_high - weekly_low) * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe
    r3_weekly_aligned = align_htf_to_ltf(prices, df_1w, r3_weekly)
    s3_weekly_aligned = align_htf_to_ltf(prices, df_1w, s3_weekly)
    
    # Weekly EMA 34 for trend filter
    ema34_1w = pd.Series(weekly_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume filter: volume > 1.5x 20-period average (vectorized)
    vol_ma20 = np.zeros_like(volume)
    vol_ma20[:20] = np.cumsum(volume[:20]) / np.arange(1, 21)
    for i in range(20, len(volume)):
        vol_ma20[i] = vol_ma20[i-1] + (volume[i] - volume[i-20]) / 20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(r3_weekly_aligned[i]) or np.isnan(s3_weekly_aligned[i]) or 
            np.isnan(ema34_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Close > R3 AND Volume > 1.5x MA AND Price > Weekly EMA34
            if (close[i] > r3_weekly_aligned[i] and 
                volume[i] > 1.5 * vol_ma20[i] and 
                close[i] > ema34_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Close < S3 AND Volume > 1.5x MA AND Price < Weekly EMA34
            elif (close[i] < s3_weekly_aligned[i] and 
                  volume[i] > 1.5 * vol_ma20[i] and 
                  close[i] < ema34_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close < S3 (reversal) OR Volume drops below average
            if (close[i] < s3_weekly_aligned[i] or volume[i] < 0.5 * vol_ma20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: Close > R3 (reversal) OR Volume drops below average
            if (close[i] > r3_weekly_aligned[i] or volume[i] < 0.5 * vol_ma20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals