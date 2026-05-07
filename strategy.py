#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_1dTrend_Volume
Hypothesis: Enter long/short when price breaks above/below daily Camarilla R1/S1 levels on 4h timeframe with volume confirmation (>2x 20-bar average) and in direction of daily EMA34 trend. Exit when price crosses opposite Camarilla level or volume drops below average. Designed for low trade frequency (~20-40/year) to minimize flood. Works in both bull and bear markets by aligning with daily trend.
"""

name = "4h_Camarilla_R1S1_Breakout_1dTrend_Volume"
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
    
    # Get daily data for trend filter and pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Previous day's OHLC for Camarilla calculation (based on completed daily bar)
    prev_daily_high = df_1d['high'].shift(1).values
    prev_daily_low = df_1d['low'].shift(1).values
    prev_daily_close = df_1d['close'].shift(1).values
    
    # Align previous day's OHLC to 4h timeframe
    prev_daily_high_aligned = align_htf_to_ltf(prices, df_1d, prev_daily_high)
    prev_daily_low_aligned = align_htf_to_ltf(prices, df_1d, prev_daily_low)
    prev_daily_close_aligned = align_htf_to_ltf(prices, df_1d, prev_daily_close)
    
    # Calculate Camarilla levels: R1 and S1
    camarilla_range = prev_daily_high_aligned - prev_daily_low_aligned
    r1 = prev_daily_close_aligned + camarilla_range * 1.1 / 12
    s1 = prev_daily_close_aligned - camarilla_range * 1.1 / 12
    
    # Volume confirmation: 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.divide(volume, vol_ma20, out=np.zeros_like(volume), where=vol_ma20!=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Warmup for volume MA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1[i]) or np.isnan(s1[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(ema_34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Daily trend determination
        daily_close_aligned = align_htf_to_ltf(prices, df_1d, df_1d['close'].values)
        if np.isnan(daily_close_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        daily_trend_up = daily_close_aligned[i] > ema_34_1d_aligned[i]
        daily_trend_down = daily_close_aligned[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1, volume spike, daily trend up
            if (close[i] > r1[i] and 
                vol_ratio[i] > 2.0 and 
                daily_trend_up):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1, volume spike, daily trend down
            elif (close[i] < s1[i] and 
                  vol_ratio[i] > 2.0 and 
                  daily_trend_down):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below S1 or volume drops below average
            if (close[i] < s1[i] or vol_ratio[i] < 1.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above R1 or volume drops below average
            if (close[i] > r1[i] or vol_ratio[i] < 1.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals