#!/usr/bin/env python3
"""
1d EMA Trend + 1w Weekly Close Filter + Volume Confirmation
Long: EMA(50) rising + weekly close above EMA(20) + volume > 1.5x average
Short: EMA(50) falling + weekly close below EMA(20) + volume > 1.5x average
Exit: Opposite EMA direction
Designed to capture multi-day trends with weekly trend confirmation and volume filter.
Target: 20-60 total trades over 4 years (5-15/year)
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
    
    # Get 1d data for EMA calculation
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(50) on daily
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate EMA(20) on daily for weekly filter
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Get 1w data for weekly close filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA(20) on weekly
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align weekly EMA to daily timeframe
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate average volume for volume spike filter
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = max(50, 20)  # need EMA50 and volume avg
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1d[i]) or np.isnan(ema_20_1d[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_avg_val = vol_avg[i]
        ema50_val = ema_50_1d[i]
        ema50_prev = ema_50_1d[i-1] if i > 0 else ema50_val
        ema20_1d_val = ema_20_1d[i]
        ema20_1w_val = ema_20_1w_aligned[i]
        
        if position == 0:
            # Long: EMA50 rising + daily close above EMA20 + weekly close above weekly EMA20 + volume spike
            if ema50_val > ema50_prev and price > ema20_1d_val and close_1d[i] > ema20_1w_val and vol > 1.5 * vol_avg_val:
                signals[i] = 0.25
                position = 1
            # Short: EMA50 falling + daily close below EMA20 + weekly close below weekly EMA20 + volume spike
            elif ema50_val < ema50_prev and price < ema20_1d_val and close_1d[i] < ema20_1w_val and vol > 1.5 * vol_avg_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: EMA50 falling or daily close below EMA20
            if ema50_val < ema50_prev or price < ema20_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: EMA50 rising or daily close above EMA20
            if ema50_val > ema50_prev or price > ema20_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_EMA_Trend_WeeklyCloseFilter_Volume"
timeframe = "1d"
leverage = 1.0