#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike
Hypothesis: Uses Camarilla pivot levels from 1d for S3/R3 breakout with volume spike confirmation and 1d EMA trend filter.
Enters long when price breaks above R3 with volume > 2x average and price above 1d EMA50.
Enters short when price breaks below S3 with volume > 2x average and price below 1d EMA50.
Designed for low trade frequency (20-40/year) with clear breakout logic, works in trending markets and avoids range-bound conditions.
"""

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Camarilla pivot levels for each day
    # R4 = C + ((H-L)*1.1/2)
    # R3 = C + ((H-L)*1.1/4)
    # S3 = C - ((H-L)*1.1/4)
    # S4 = C - ((H-L)*1.1/2)
    camarilla_r3 = df_1d['close'] + (df_1d['high'] - df_1d['low']) * 1.1 / 4
    camarilla_s3 = df_1d['close'] - (df_1d['high'] - df_1d['low']) * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3.values)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3.values)
    
    # Volume confirmation: 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.divide(volume, vol_ma20, out=np.zeros_like(volume), where=vol_ma20!=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(ema_50_1d_aligned[i])):
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
            
        daily_trend_up = daily_close_aligned[i] > ema_50_1d_aligned[i]
        daily_trend_down = daily_close_aligned[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above R3, volume spike, price above EMA50
            if (close[i] > camarilla_r3_aligned[i] and 
                vol_ratio[i] > 2.0 and 
                daily_trend_up):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3, volume spike, price below EMA50
            elif (close[i] < camarilla_s3_aligned[i] and 
                  vol_ratio[i] > 2.0 and 
                  daily_trend_down):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price falls below S3 or trend changes
            if close[i] < camarilla_s3_aligned[i] or not daily_trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price rises above R3 or trend changes
            if close[i] > camarilla_r3_aligned[i] or not daily_trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals