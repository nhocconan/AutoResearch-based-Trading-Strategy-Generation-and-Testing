#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_camarilla_breakout_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for previous day (H4/L4)
    camarilla_high = np.full(len(close_1d), np.nan)
    camarilla_low = np.full(len(close_1d), np.nan)
    
    for i in range(1, len(close_1d)):
        H = high_1d[i-1]
        L = low_1d[i-1]
        C = close_1d[i-1]
        camarilla_high[i] = C + ((H - L) * 1.1 / 2)
        camarilla_low[i] = C - ((H - L) * 1.1 / 2)
    
    # Align to 12h timeframe
    camarilla_high_aligned = align_htf_to_ltf(prices, df_1d, camarilla_high)
    camarilla_low_aligned = align_htf_to_ltf(prices, df_1d, camarilla_low)
    
    # Volume filter: current volume > 20-period average (on 12h data)
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Pre-calculate pivot points for exit
    pivot_point = np.full(len(close_1d), np.nan)
    for j in range(1, len(close_1d)):
        H = high_1d[j-1]
        L = low_1d[j-1]
        C = close_1d[j-1]
        pivot_point[j] = (H + L + C) / 3
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_point)
    
    for i in range(20, n):  # warmup for volume filter
        # Skip if not ready
        if (np.isnan(camarilla_high_aligned[i]) or np.isnan(camarilla_low_aligned[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(volume_ok[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Breakout conditions with volume confirmation
        breakout_up = close[i] > camarilla_high_aligned[i]
        breakout_down = close[i] < camarilla_low_aligned[i]
        vol_ok = volume_ok[i]
        
        # Exit when price returns to the Camarilla pivot (close of previous day)
        exit_long = close[i] < pivot_aligned[i]
        exit_short = close[i] > pivot_aligned[i]
        
        # Execute trades
        if breakout_up and vol_ok and position != 1:
            position = 1
            signals[i] = 0.25
        elif breakout_down and vol_ok and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals