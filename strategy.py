#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_1dTrend_Volume
Hypothesis: 4-hour breakouts above Camarilla R1 or below S1 with 1-day EMA34 trend filter and volume confirmation.
Combines intraday support/resistance (Camarilla) with daily trend filter to reduce whipsaw and improve performance in both bull and bear markets.
Target: 20-50 trades/year with strict entry conditions to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1-day EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA34 with proper smoothing
    ema34_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 34:
        ema34_1d[33] = np.mean(close_1d[0:34])
        alpha = 2 / (34 + 1)
        for i in range(34, len(close_1d)):
            ema34_1d[i] = close_1d[i] * alpha + ema34_1d[i-1] * (1 - alpha)
    
    # Align 1-day EMA34 to 4h timeframe
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate true daily Camarilla levels from previous day
    # We need actual daily OHLC - use 1d data from htF
    daily_open = df_1d['open'].values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    camarilla_r1 = np.full(n, np.nan)
    camarilla_s1 = np.full(n, np.nan)
    
    # Map each 4h bar to its corresponding previous day
    # Create mapping from 4h index to 1d index
    day_index = np.zeros(n, dtype=int)
    day_index[:] = -1
    
    for i in range(n):
        # Find the 1d bar that contains this 4h bar's open_time
        # Since we don't have the actual datetime, we'll use index mapping
        # 6 four-hour bars per day
        day_index[i] = i // 6
    
    for i in range(n):
        day_idx = day_index[i]
        if day_idx >= 1 and day_idx < len(daily_close):  # Ensure we have previous day
            prev_high = daily_high[day_idx - 1]
            prev_low = daily_low[day_idx - 1]
            prev_close = daily_close[day_idx - 1]
            range_val = prev_high - prev_low
            if range_val > 0:  # Avoid division by zero
                camarilla_r1[i] = prev_close + range_val * 1.1 / 12
                camarilla_s1[i] = prev_close - range_val * 1.1 / 12
    
    # Volume spike: current volume > 1.5 x 20-period average (more reasonable threshold)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20, 1)  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_r1[i]) or np.isnan(camarilla_s1[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above Camarilla R1 with volume spike and 1-day uptrend
            if (close[i] > camarilla_r1[i] and vol_spike[i] and 
                close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below Camarilla S1 with volume spike and 1-day downtrend
            elif (close[i] < camarilla_s1[i] and vol_spike[i] and 
                  close[i] < ema34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: close below Camarilla S1 or 1-day trend turns down
            if (close[i] < camarilla_s1[i] or close[i] < ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: close above Camarilla R1 or 1-day trend turns up
            if (close[i] > camarilla_r1[i] or close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0