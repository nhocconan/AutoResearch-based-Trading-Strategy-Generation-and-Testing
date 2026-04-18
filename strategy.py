#!/usr/bin/env python3
"""
4h_Pivot_R1S1_Breakout_1dEMA50_Volume
Hypothesis: 4-hour breakouts above 1d Camarilla R1 or below S1 with 1-day EMA50 trend filter and volume confirmation.
Uses actual daily OHLC for precise Camarilla levels, EMA50 filters trend direction, volume confirms breakout strength.
Designed for low trade frequency (target: 20-50/year) with strong performance in both bull and bear markets.
"""

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
    
    # Calculate 1-day EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 with proper smoothing
    ema50_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema50_1d[49] = np.mean(close_1d[0:50])
        alpha = 2 / (50 + 1)
        for i in range(50, len(close_1d)):
            ema50_1d[i] = close_1d[i] * alpha + ema50_1d[i-1] * (1 - alpha)
    
    # Align 1-day EMA50 to 4h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Camarilla levels from previous day's OHLC
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_r1 = np.full(n, np.nan)
    camarilla_s1 = np.full(n, np.nan)
    
    # Get daily OHLC from 1d data and align to 4h
    daily_open = df_1d['open'].values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Align daily values to 4h timeframe (use previous day's values for today's levels)
    daily_open_aligned = align_htf_to_ltf(prices, df_1d, daily_open)
    daily_high_aligned = align_htf_to_ltf(prices, df_1d, daily_high)
    daily_low_aligned = align_htf_to_ltf(prices, df_1d, daily_low)
    daily_close_aligned = align_htf_to_ltf(prices, df_1d, daily_close)
    
    # Calculate Camarilla levels using previous day's OHLC
    for i in range(n):
        if (np.isnan(daily_open_aligned[i]) or np.isnan(daily_high_aligned[i]) or 
            np.isnan(daily_low_aligned[i]) or np.isnan(daily_close_aligned[i])):
            continue
        
        # Use previous day's values (shifted by 1 to avoid look-ahead)
        if i > 0:
            prev_high = daily_high_aligned[i-1]
            prev_low = daily_low_aligned[i-1]
            prev_close = daily_close_aligned[i-1]
            range_val = prev_high - prev_low
            camarilla_r1[i] = prev_close + range_val * 1.1 / 12
            camarilla_s1[i] = prev_close - range_val * 1.1 / 12
    
    # Volume spike: current volume > 2.0 x 20-period average (stricter to reduce trades)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20, 1)  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_r1[i]) or np.isnan(camarilla_s1[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above Camarilla R1 with volume spike and 1d uptrend
            if (close[i] > camarilla_r1[i] and vol_spike[i] and 
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below Camarilla S1 with volume spike and 1d downtrend
            elif (close[i] < camarilla_s1[i] and vol_spike[i] and 
                  close[i] < ema50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: close below Camarilla S1 or 1d trend turns down
            if (close[i] < camarilla_s1[i] or close[i] < ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: close above Camarilla R1 or 1d trend turns up
            if (close[i] > camarilla_r1[i] or close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Pivot_R1S1_Breakout_1dEMA50_Volume"
timeframe = "4h"
leverage = 1.0