#!/usr/bin/env python3
"""
4h_Pivot_R1_S1_Breakout_Volume_Confirm_v1
Hypothesis: Trade breakouts of Camarilla pivot levels R1 and S1 on 4h with volume confirmation and 12h trend filter. 
Long when price breaks above R1 with volume > 1.5x average and 12h EMA34 uptrend. 
Short when price breaks below S1 with volume > 1.5x average and 12h EMA34 downtrend.
Exit when price returns to the pivot point (P) or reverses with opposite volume signal.
Designed for low trade frequency (<40/year) to minimize fee drag while capturing meaningful breakouts in both bull and bear markets.
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
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for each day
    pivot = (high_1d + low_1d + close_1d) / 3
    range_ = high_1d - low_1d
    r1 = close_1d + range_ * 1.1 / 12
    s1 = close_1d - range_ * 1.1 / 12
    
    # Align pivot levels to 4h timeframe (previous day's levels)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate EMA34 on 12h
    ema_period = 34
    ema_12h = np.full_like(close_12h, np.nan)
    if len(close_12h) >= ema_period:
        ema_12h[ema_period-1] = np.mean(close_12h[:ema_period])
        for i in range(ema_period, len(close_12h)):
            ema_12h[i] = (close_12h[i] * 2 / (ema_period + 1)) + (ema_12h[i-1] * (ema_period - 1) / (ema_period + 1))
    
    # Align EMA34 to 4h timeframe
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, vol_period)  # Need sufficient data for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(ema_12h_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above R1 + volume + 12h uptrend
            if close[i] > r1_aligned[i] and vol_confirm and ema_12h_aligned[i] > ema_12h_aligned[i-1]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + volume + 12h downtrend
            elif close[i] < s1_aligned[i] and vol_confirm and ema_12h_aligned[i] < ema_12h_aligned[i-1]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to pivot or reverses with volume
            if close[i] <= pivot_aligned[i] or (close[i] < close[i-1] and vol_confirm):
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to pivot or reverses with volume
            if close[i] >= pivot_aligned[i] or (close[i] > close[i-1] and vol_confirm):
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Pivot_R1_S1_Breakout_Volume_Confirm_v1"
timeframe = "4h"
leverage = 1.0