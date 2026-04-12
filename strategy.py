#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_camarilla_volume_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Camarilla calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 5:
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla levels for previous 12h period
    # Camarilla: H4 = C + ((H-L) * 1.1/2), L4 = C - ((H-L) * 1.1/2)
    camarilla_high = np.zeros(len(close_12h))
    camarilla_low = np.zeros(len(close_12h))
    
    for i in range(1, len(close_12h)):
        H = high_12h[i-1]
        L = low_12h[i-1]
        C = close_12h[i-1]
        camarilla_high[i] = C + ((H - L) * 1.1 / 2)
        camarilla_low[i] = C - ((H - L) * 1.1 / 2)
    
    # Align to 4h timeframe
    camarilla_high_aligned = align_htf_to_ltf(prices, df_12h, camarilla_high)
    camarilla_low_aligned = align_htf_to_ltf(prices, df_12h, camarilla_low)
    
    # Volume filter: current volume > 20-period average (on 4h data)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # warmup for volume filter
        # Skip if not ready
        if (np.isnan(camarilla_high_aligned[i]) or np.isnan(camarilla_low_aligned[i]) or 
            np.isnan(volume_ok[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Breakout conditions with volume confirmation
        breakout_up = close[i] > camarilla_high_aligned[i]
        breakout_down = close[i] < camarilla_low_aligned[i]
        
        # Volume confirmation
        vol_ok = volume_ok[i]
        
        # Entry signals
        long_signal = breakout_up and vol_ok
        short_signal = breakout_down and vol_ok
        
        # Exit when price returns to the Camarilla pivot (close of previous 12h period)
        # Calculate pivot point: (H + L + C) / 3
        pivot_point = np.zeros(len(close_12h))
        for j in range(1, len(close_12h)):
            H = high_12h[j-1]
            L = low_12h[j-1]
            C = close_12h[j-1]
            pivot_point[j] = (H + L + C) / 3
        pivot_aligned = align_htf_to_ltf(prices, df_12h, pivot_point)
        
        exit_long = close[i] < pivot_aligned[i]
        exit_short = close[i] > pivot_aligned[i]
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
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