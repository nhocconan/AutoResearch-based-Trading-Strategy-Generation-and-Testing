#!/usr/bin/env python3
"""
1h_Camarilla_4hTrend_Volume
Hypothesis: 1-hour breakouts above/below daily Camarilla R1/S1 with 4-hour EMA34 trend filter and volume confirmation.
Uses 4h EMA34 for directional bias (works in both bull/bear markets), daily Camarilla for precise S/R levels,
and volume spike for breakout confirmation. Targets 15-37 trades/year by using 4h trend filter to reduce noise.
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
    
    # Calculate 4-hour EMA34 trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate EMA34 with proper smoothing
    ema34_4h = np.full(len(close_4h), np.nan)
    if len(close_4h) >= 34:
        ema34_4h[33] = np.mean(close_4h[0:34])
        alpha = 2 / (34 + 1)
        for i in range(34, len(close_4h)):
            ema34_4h[i] = close_4h[i] * alpha + ema34_4h[i-1] * (1 - alpha)
    
    # Align 4-hour EMA34 to 1h timeframe
    ema34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h)
    
    # Calculate daily Camarilla levels from previous day
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_r1 = np.full(n, np.nan)
    camarilla_s1 = np.full(n, np.nan)
    
    # Get daily data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Map daily levels to 1h bars
    daily_open = df_1d['open'].values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Calculate daily Camarilla levels
    camarilla_r1_daily = np.full(len(df_1d), np.nan)
    camarilla_s1_daily = np.full(len(df_1d), np.nan)
    
    for i in range(1, len(df_1d)):
        range_val = daily_high[i-1] - daily_low[i-1]
        camarilla_r1_daily[i] = daily_close[i-1] + range_val * 1.1 / 12
        camarilla_s1_daily[i] = daily_close[i-1] - range_val * 1.1 / 12
    
    # Align daily Camarilla levels to 1h timeframe
    camarilla_r1 = align_htf_to_ltf(prices, df_1d, camarilla_r1_daily)
    camarilla_s1 = align_htf_to_ltf(prices, df_1d, camarilla_s1_daily)
    
    # Volume spike: current volume > 2.0 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 2.0)
    
    # Session filter: 08:00-20:00 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20, 1)  # Ensure indicators ready
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_r1[i]) or np.isnan(camarilla_s1[i]) or 
            np.isnan(ema34_4h_aligned[i]) or np.isnan(vol_ma[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above daily Camarilla R1 with volume spike and 4h uptrend
            if (close[i] > camarilla_r1[i] and vol_spike[i] and 
                close[i] > ema34_4h_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short: break below daily Camarilla S1 with volume spike and 4h downtrend
            elif (close[i] < camarilla_s1[i] and vol_spike[i] and 
                  close[i] < ema34_4h_aligned[i]):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: close below daily Camarilla S1 or 4h trend turns down
            if (close[i] < camarilla_s1[i] or close[i] < ema34_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: close above daily Camarilla R1 or 4h trend turns up
            if (close[i] > camarilla_r1[i] or close[i] > ema34_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0