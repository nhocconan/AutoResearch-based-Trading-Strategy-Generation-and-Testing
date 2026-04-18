#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_1dTrend_Volume
Hypothesis: 12-hour breakouts above Camarilla R1 or below S1 with 1-day EMA34 trend filter and volume confirmation.
Camarilla levels provide precise daily support/resistance, EMA34 filters trend direction, volume confirms breakout strength.
Designed for low trade frequency (target: 12-37/year) with strong performance in both bull and bear markets.
"""

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
    
    # Align 1-day EMA34 to 12h timeframe
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Camarilla levels from previous day
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_r1 = np.full(n, np.nan)
    camarilla_s1 = np.full(n, np.nan)
    
    for i in range(1, n):
        # Use previous day's OHLC (using daily data from 1d aggregation)
        # For 12h data, we need to look back to previous day's values
        if i >= 2:  # Need at least 2 bars for previous day
            # Get previous day's high, low, close from 1d data
            # Since we're using 12h timeframe, we need to map to daily bars
            prev_day_idx = i - 2  # Previous day in 12h terms (2 bars = 1 day)
            if prev_day_idx < len(df_1d):
                prev_high = df_1d['high'].iloc[prev_day_idx]
                prev_low = df_1d['low'].iloc[prev_day_idx]
                prev_close = df_1d['close'].iloc[prev_day_idx]
                range_val = prev_high - prev_low
                camarilla_r1[i] = prev_close + range_val * 1.1 / 12
                camarilla_s1[i] = prev_close - range_val * 1.1 / 12
    
    # Volume spike: current volume > 1.8 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20, 2)  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_r1[i]) or np.isnan(camarilla_s1[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above Camarilla R1 with volume spike and 1d uptrend
            if (close[i] > camarilla_r1[i] and vol_spike[i] and 
                close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below Camarilla S1 with volume spike and 1d downtrend
            elif (close[i] < camarilla_s1[i] and vol_spike[i] and 
                  close[i] < ema34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: close below Camarilla S1 or 1d trend turns down
            if (close[i] < camarilla_s1[i] or close[i] < ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: close above Camarilla R1 or 1d trend turns up
            if (close[i] > camarilla_r1[i] or close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0