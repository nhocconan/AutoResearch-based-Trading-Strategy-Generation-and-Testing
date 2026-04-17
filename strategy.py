#!/usr/bin/env python3
"""
6h Weekly Pivot Breakout with Volume and Daily Trend Filter
Long: Price breaks above weekly pivot R1 + volume > 1.5x 6h volume MA + price > daily EMA50
Short: Price breaks below weekly pivot S1 + volume > 1.5x 6h volume MA + price < daily EMA50
Exit: Opposite break of weekly pivot level (S1 for long, R1 for short)
Uses weekly pivot from 1w data and daily EMA50 for trend filter
Target: 12-37 trades/year per symbol (50-150 total over 4 years)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    # Calculate weekly pivot points (using prior week's OHLC)
    weekly_high = df_1w['high'].shift(1)  # Prior week's high
    weekly_low = df_1w['low'].shift(1)    # Prior week's low
    weekly_close = df_1w['close'].shift(1) # Prior week's close
    
    # Weekly pivot point and support/resistance levels
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_r1 = 2 * weekly_pivot - weekly_low
    weekly_s1 = 2 * weekly_pivot - weekly_high
    
    # Get daily EMA for trend filter
    df_1d = get_htf_data(prices, '1d')
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get 6h volume moving average (20-period for confirmation)
    df_6h = get_htf_data(prices, '6h')
    volume_ma_20 = pd.Series(df_6h['volume']).rolling(window=20, min_periods=20).mean()
    volume_ma_20_6h = align_htf_to_ltf(prices, df_6h, volume_ma_20.values)
    
    # Align weekly pivot levels to 6h timeframe
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1.values)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1.values)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot.values)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    entry_price = 0.0
    
    start_idx = 100  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(weekly_r1_aligned[i]) or np.isnan(weekly_s1_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_ma_20_6h[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20_6h[i]
        
        if position == 0:
            # Long: break above weekly R1 + volume + daily trend
            if price > weekly_r1_aligned[i] and vol > 1.5 * vol_ma and price > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: break below weekly S1 + volume + daily trend
            elif price < weekly_s1_aligned[i] and vol > 1.5 * vol_ma and price < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: break below weekly S1
            if price < weekly_s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: break above weekly R1
            if price > weekly_r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_R1S1_Volume_DailyTrend"
timeframe = "6h"
leverage = 1.0