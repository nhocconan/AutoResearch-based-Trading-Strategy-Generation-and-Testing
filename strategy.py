#!/usr/bin/env python3
"""
1D_Camarilla_R3S3_Breakout_1WTrend_VolumeFilter_v1
Hypothesis: Use weekly Camarilla R3/S3 levels for breakout signals with 1d EMA(34) trend filter and volume confirmation.
Long when price breaks above weekly R3 and close > 1d EMA(34); Short when price breaks below weekly S3 and close < 1d EMA(34).
Volume confirmation: current volume > 1.3x 20-period average volume.
This targets breakouts in trending markets while filtering ranges, designed for low trade frequency (<15/year) to minimize fee drag.
"""
name = "1D_Camarilla_R3S3_Breakout_1WTrend_VolumeFilter_v1"
timeframe = "1d"
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
    
    # Get weekly data for Camarilla levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly Camarilla levels: R3, S3
    # Camarilla: R3 = close + (high - low) * 1.1/4, S3 = close - (high - low) * 1.1/4
    weekly_range = df_1w['high'] - df_1w['low']
    camarilla_r3 = df_1w['close'] + weekly_range * 1.1 / 4
    camarilla_s3 = df_1w['close'] - weekly_range * 1.1 / 4
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3.values)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3.values)
    
    # Get daily data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend
    ema_34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Volume filter: current volume > 1.3 * 20-period average volume
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_exit = 0  # bars since last exit to prevent overtrading
    
    start_idx = max(34, 20)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        bars_since_exit += 1
        
        # Skip if any data is not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            continue
        
        if position == 0:
            # Minimum 10 days between trades to reduce frequency
            if bars_since_exit < 10:
                continue
                
            # Long: price breaks above weekly R3 + close > 1d EMA34 + volume filter
            if (close[i] > camarilla_r3_aligned[i] and 
                close[i] > ema_34_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_exit = 0
            # Short: price breaks below weekly S3 + close < 1d EMA34 + volume filter
            elif (close[i] < camarilla_s3_aligned[i] and 
                  close[i] < ema_34_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_exit = 0
        elif position != 0:
            # Exit: price returns to opposite Camarilla level or trend reversal
            if position == 1 and (close[i] < camarilla_s3_aligned[i] or close[i] < ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            elif position == -1 and (close[i] > camarilla_r3_aligned[i] or close[i] > ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals