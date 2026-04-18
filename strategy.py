#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_Volume_Trend
Hypothesis: Trade breakouts of Camarilla R1/S1 levels on 12h timeframe with volume confirmation and 1d EMA trend filter. Camarilla levels act as intraday support/resistance; breakouts above R1 or below S1 with volume indicate institutional participation. The 1d EMA filter ensures trades align with higher timeframe trend, reducing false signals in choppy markets. Designed for 12h timeframe to target 12-37 trades/year (50-150 total over 4 years) to minimize fee drag. Works in both bull and bear markets by filtering counter-trend trades.
"""

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
    
    # Get 12h data for Camarilla calculation
    df_12h = get_htf_data(prices, '12h')
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # 12h calculations
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Previous 12h bar's OHLC (completed bar)
    prev_high_12h = np.roll(high_12h, 1)
    prev_low_12h = np.roll(low_12h, 1)
    prev_close_12h = np.roll(close_12h, 1)
    prev_high_12h[0] = high_12h[0]
    prev_low_12h[0] = low_12h[0]
    prev_close_12h[0] = close_12h[0]
    
    # Camarilla levels calculation
    # Range = (high - low) of previous period
    range_12h = prev_high_12h - prev_low_12h
    
    # Camarilla resistance and support levels
    # R1 = close + (range * 1.1/12)
    # S1 = close - (range * 1.1/12)
    camarilla_multiplier = 1.1 / 12
    r1_12h = prev_close_12h + (range_12h * camarilla_multiplier)
    s1_12h = prev_close_12h - (range_12h * camarilla_multiplier)
    
    # 1d EMA trend filter
    close_1d = df_1d['close'].values
    ema_period = 34
    if len(close_1d) >= ema_period:
        ema_1d = np.zeros_like(close_1d)
        ema_1d[ema_period-1] = np.mean(close_1d[:ema_period])
        for i in range(ema_period, len(close_1d)):
            ema_1d[i] = (close_1d[i] * 2 / (ema_period + 1)) + (ema_1d[i-1] * (ema_period - 1) / (ema_period + 1))
    else:
        ema_1d = np.full_like(close_1d, np.nan)
    
    # Align higher timeframe data to 12h
    r1_12h_aligned = align_htf_to_ltf(prices, df_12h, r1_12h)
    s1_12h_aligned = align_htf_to_ltf(prices, df_12h, s1_12h)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = np.zeros_like(volume)
    vol_period = 20
    for i in range(vol_period, len(volume)):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, vol_period, ema_period)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_12h_aligned[i]) or np.isnan(s1_12h_aligned[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume and above 1d EMA
            if close[i] > r1_12h_aligned[i] and vol_confirm and close[i] > ema_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume and below 1d EMA
            elif close[i] < s1_12h_aligned[i] and vol_confirm and close[i] < ema_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price closes below S1 (reverse signal) or below 1d EMA
            if close[i] < s1_12h_aligned[i] or close[i] < ema_1d_aligned[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price closes above R1 (reverse signal) or above 1d EMA
            if close[i] > r1_12h_aligned[i] or close[i] > ema_1d_aligned[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_Volume_Trend"
timeframe = "12h"
leverage = 1.0