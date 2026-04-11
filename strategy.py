#!/usr/bin/env python3
# 6h_1d_ema_crossover_volume_filter_v1
# Strategy: 6-hour EMA(9/21) crossover with volume confirmation and 1-day EMA(50) trend filter
# Timeframe: 6h
# Leverage: 1.0
# Hypothesis: EMA crossovers capture momentum; volume filters false signals; daily EMA50 aligns with longer-term trend.
# Works in bull markets by capturing uptrend continuations; in bear markets by catching downtrend reversals.
# Uses tight entry conditions (volume + trend filter) to limit trades (~20-40/year) and avoid fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_ema_crossover_volume_filter_v1"
timeframe = "6h"
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
    
    # Load daily data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Daily EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 6h EMA(9) and EMA(21) for crossover
    close_series = pd.Series(close)
    ema9 = close_series.ewm(span=9, adjust=False, min_periods=9).mean().values
    ema21 = close_series.ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # 6h Volume confirmation: current volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(21, n):  # Start after EMA21 warmup
        # Skip if any required data is invalid
        if (np.isnan(ema9[i]) or np.isnan(ema21[i]) or 
            np.isnan(ema9[i-1]) or np.isnan(ema21[i-1]) or
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # EMA crossovers
        ema_cross_up = ema9[i-1] <= ema21[i-1] and ema9[i] > ema21[i]
        ema_cross_down = ema9[i-1] >= ema21[i-1] and ema9[i] < ema21[i]
        
        # Trend filter: price above/below daily EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Entry logic: EMA cross + volume + trend alignment
        if ema_cross_up and vol_confirm[i] and uptrend and position != 1:
            position = 1
            signals[i] = 0.25
        elif ema_cross_down and vol_confirm[i] and downtrend and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: opposite EMA cross with volume confirmation
        elif position == 1 and ema_cross_down and vol_confirm[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and ema_cross_up and vol_confirm[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals