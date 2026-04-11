#!/usr/bin/env python3
"""
1d_1w_ema_cross_volume_trend_v1
Strategy: 1d EMA20/50 cross with volume confirmation and 1w trend filter
Timeframe: 1d
Leverage: 1.0
Hypothesis: Uses daily EMA20/50 cross for trend changes, confirmed by volume > 1.5x average and filtered by weekly EMA50 trend. Designed to capture medium-term trends in both bull and bear markets by entering on EMA cross with volume confirmation in the direction of the weekly trend. Weekly trend filter prevents counter-trend trades in strong trends, reducing whipsaw. Target: 20-60 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_ema_cross_volume_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load higher timeframe data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1d EMA20 and EMA50 for crossover
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume average (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_avg * 1.5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_20[i]) or np.isnan(ema_50[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # EMA cross conditions
        ema_cross_up = ema_20[i] > ema_50[i] and ema_20[i-1] <= ema_50[i-1]
        ema_cross_down = ema_20[i] < ema_50[i] and ema_20[i-1] >= ema_50[i-1]
        
        # Volume confirmation
        vol_confirmed = volume[i] > vol_threshold[i]
        
        # Weekly trend filter
        uptrend_1w = close[i] > ema_50_1w_aligned[i]
        downtrend_1w = close[i] < ema_50_1w_aligned[i]
        
        # Exit conditions: opposite EMA cross
        exit_long = position == 1 and ema_cross_down
        exit_short = position == -1 and ema_cross_up
        
        # Trading logic
        if ema_cross_up and vol_confirmed and uptrend_1w and position != 1:
            position = 1
            signals[i] = 0.25
        elif ema_cross_down and vol_confirmed and downtrend_1w and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals