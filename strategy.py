#!/usr/bin/env python3
"""
4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_Dyn
Hypothesis: Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike (>2x average). Designed for low trade frequency (19-50/year) to avoid fee drag. Works in both bull and bear markets by aligning with daily trend direction. Uses discrete position sizing (0.30) to minimize churn.
"""

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_Dyn"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate previous day's Camarilla levels (R3, S3)
    # Using previous day's OHLC (already completed due to get_htf_data alignment)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla formulas
    rang = prev_high - prev_low
    r3 = prev_close + rang * 1.1 / 2
    s3 = prev_close - rang * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume confirmation: 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.divide(volume, vol_ma20, out=np.zeros_like(volume), where=vol_ma20!=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Warmup for EMA34
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(ema_34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine daily trend using aligned close
        daily_close_aligned = align_htf_to_ltf(prices, df_1d, df_1d['close'].values)
        if np.isnan(daily_close_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        daily_trend_up = daily_close_aligned[i] > ema_34_1d_aligned[i]
        daily_trend_down = daily_close_aligned[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above R3, volume spike, price above EMA34
            if (close[i] > r3_aligned[i] and 
                vol_ratio[i] > 2.0 and 
                daily_trend_up):
                signals[i] = 0.30
                position = 1
            # Short: price breaks below S3, volume spike, price below EMA34
            elif (close[i] < s3_aligned[i] and 
                  vol_ratio[i] > 2.0 and 
                  daily_trend_down):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price falls below S3 or trend changes
            if close[i] < s3_aligned[i] or not daily_trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price rises above R3 or trend changes
            if close[i] > r3_aligned[i] or not daily_trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals