#!/usr/bin/env python3
name = "1h_Camarilla_R1_S1_Breakout_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0

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
    
    # Get 4h data for trend filter (EMA50)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 4h data for Camarilla levels (R1, S1) and volume filter
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Camarilla: R1 = close + (high - low) * 1.1 / 12, S1 = close - (high - low) * 1.1 / 12
    r1 = close_4h + (high_4h - low_4h) * 1.1 / 12
    s1 = close_4h - (high_4h - low_4h) * 1.1 / 12
    
    r1_aligned = align_htf_to_ltf(prices, df_4h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_4h, s1)
    
    # Volume filter: current 4h volume > 20-period average volume
    vol_4h = df_4h['volume'].values
    vol_avg = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
    volume_filter = vol_4h > vol_avg
    volume_filter_aligned = align_htf_to_ltf(prices, df_4h, volume_filter)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(volume_filter_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if outside session
        if not session_filter[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above 4h EMA50 (uptrend), 1h close above 4h R1, volume confirmation
            if (close[i] > ema_50_4h_aligned[i] and 
                close[i] > r1_aligned[i] and 
                volume_filter_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short: price below 4h EMA50 (downtrend), 1h close below 4h S1, volume confirmation
            elif (close[i] < ema_50_4h_aligned[i] and 
                  close[i] < s1_aligned[i] and 
                  volume_filter_aligned[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price crosses below 4h EMA50 (trend change)
            if close[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price crosses above 4h EMA50 (trend change)
            if close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals