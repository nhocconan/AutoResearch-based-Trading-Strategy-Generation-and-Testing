#!/usr/bin/env python3
"""
12h_Camarilla_R3S3_Breakout_1dTrend_Volume_v2
Hypothesis: Use daily Camarilla R3/S3 levels for breakout entries, filtered by daily EMA34 trend and volume confirmation. Long when price breaks above R3 in daily uptrend with volume spike, short when breaks below S3 in daily downtrend with volume spike. Exit on opposite level touch. Designed for 12h to capture multi-day swings with low frequency (target 12-37 trades/year).
"""

name = "12h_Camarilla_R3S3_Breakout_1dTrend_Volume_v2"
timeframe = "12h"
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
    
    # Get daily data for Camarilla levels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate Camarilla levels (R3, S3) from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R3 and S3 (based on previous day's range)
    # R3 = close + 1.1 * (high - low) / 6
    # S3 = close - 1.1 * (high - low) / 6
    camarilla_r3 = close_1d + 1.1 * (high_1d - low_1d) / 6
    camarilla_s3 = close_1d - 1.1 * (high_1d - low_1d) / 6
    
    # Align Camarilla levels to 12h timeframe (no extra delay needed as levels are known at close)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Daily trend filter: EMA34
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: 20-period average on 12h
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.divide(volume, vol_ma20, out=np.zeros_like(volume), where=vol_ma20!=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Warmup for daily EMA34
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Daily trend determination
        daily_close_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
        if np.isnan(daily_close_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        daily_trend_up = daily_close_aligned[i] > ema_34_1d_aligned[i]
        daily_trend_down = daily_close_aligned[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above R3 with volume spike in daily uptrend
            if (close[i] > camarilla_r3_aligned[i] and
                vol_ratio[i] > 2.5 and 
                daily_trend_up):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 with volume spike in daily downtrend
            elif (close[i] < camarilla_s3_aligned[i] and 
                  vol_ratio[i] > 2.5 and 
                  daily_trend_down):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price touches or goes below S3
            if close[i] <= camarilla_s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price touches or goes above R3
            if close[i] >= camarilla_r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals