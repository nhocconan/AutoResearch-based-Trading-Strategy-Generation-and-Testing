#!/usr/bin/env python3
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
    
    # Get weekly data once for HTF context
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate weekly range for pivot calculations
    weekly_range = high_1w - low_1w
    
    # Weekly Camarilla pivot levels (based on previous week)
    camarilla_r4 = close_1w + weekly_range * 1.1 / 2
    camarilla_s4 = close_1w - weekly_range * 1.1 / 2
    
    # Daily EMA200 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align weekly Camarilla levels and daily EMA200 to daily timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r4)
    s4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s4)
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Volume filter: above average weekly volume (20-period)
    vol_ma = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1w, vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(ema_200_1d_aligned[i]) or np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions: 
        # Long: price breaks above R4 with volume and above daily EMA200
        # Short: price breaks below S4 with volume and below daily EMA200
        long_entry = (close[i] > r4_aligned[i]) and (volume[i] > vol_ma_aligned[i]) and (close[i] > ema_200_1d_aligned[i])
        short_entry = (close[i] < s4_aligned[i]) and (volume[i] > vol_ma_aligned[i]) and (close[i] < ema_200_1d_aligned[i])
        
        # Exit conditions: price returns to opposite S4/R4 levels
        long_exit = close[i] < s4_aligned[i]
        short_exit = close[i] > r4_aligned[i]
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_WeeklyCamarilla_R4S4_Breakout_EMA200"
timeframe = "1d"
leverage = 1.0