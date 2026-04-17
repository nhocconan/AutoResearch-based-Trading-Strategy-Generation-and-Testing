#!/usr/bin/env python3
"""
4h_Camarilla_Pivot_Breakout_Volume_TrendFilter
Strategy: Breakout above Camarilla R3 or below S3 with volume confirmation and trend filter.
Long: Close > R3 + volume > 1.5x avg + EMA34 > EMA144
Short: Close < S3 + volume > 1.5x avg + EMA34 < EMA144
Exit: Close crosses back below R3 (long) or above S3 (short) or trend reversal
Position size: 0.25
Designed to capture strong breakout moves with institutional interest at key levels.
Timeframe: 4h
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA34 and EMA144 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    close_series_1d = pd.Series(close_1d)
    ema34_1d = close_series_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema144_1d = close_series_1d.ewm(span=144, adjust=False, min_periods=144).mean().values
    
    # Align 1d EMAs to 4h timeframe
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    ema144_1d_aligned = align_htf_to_ltf(prices, df_1d, ema144_1d)
    
    # Calculate daily Camarilla levels from previous day's OHLC
    df_1d_full = get_htf_data(prices, '1d')
    high_1d = df_1d_full['high'].values
    low_1d = df_1d_full['low'].values
    close_1d_full = df_1d_full['close'].values
    
    # Calculate Camarilla levels for each day
    camarilla_r3 = np.full(len(close_1d_full), np.nan)
    camarilla_s3 = np.full(len(close_1d_full), np.nan)
    
    for i in range(len(close_1d_full)):
        if i >= 1:  # Need previous day's data
            prev_high = high_1d[i-1]
            prev_low = low_1d[i-1]
            prev_close = close_1d_full[i-1]
            
            if prev_high > prev_low:  # Valid range
                range_val = prev_high - prev_low
                camarilla_r3[i] = prev_close + (range_val * 1.1 / 4)  # R3 = C + 1.1*(H-L)/4
                camarilla_s3[i] = prev_close - (range_val * 1.1 / 4)  # S3 = C - 1.1*(H-L)/4
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d_full, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d_full, camarilla_s3)
    
    # Volume confirmation (20-period MA on 4h)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(144, 20)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(ema144_1d_aligned[i]) or 
            np.isnan(volume_ma20[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period average
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Trend filter: 1d EMA34 > EMA144 for long, < for short
        ema34_gt_ema144 = ema34_1d_aligned[i] > ema144_1d_aligned[i]
        ema34_lt_ema144 = ema34_1d_aligned[i] < ema144_1d_aligned[i]
        
        if position == 0:
            # Long: breakout above R3 + volume + uptrend
            if close[i] > camarilla_r3_aligned[i] and volume_filter and ema34_gt_ema144:
                signals[i] = 0.25
                position = 1
            # Short: breakdown below S3 + volume + downtrend
            elif close[i] < camarilla_s3_aligned[i] and volume_filter and ema34_lt_ema144:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses back below R3 or trend reverses
            if close[i] < camarilla_r3_aligned[i] or not ema34_gt_ema144:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses back above S3 or trend reverses
            if close[i] > camarilla_s3_aligned[i] or not ema34_lt_ema144:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_Pivot_Breakout_Volume_TrendFilter"
timeframe = "4h"
leverage = 1.0