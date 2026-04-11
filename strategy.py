#!/usr/bin/env python3
# 1h_4h_1d_ema_cross_volume_v1
# Strategy: 1h EMA(20)/EMA(50) cross with 4h trend filter and volume confirmation
# Timeframe: 1h
# Leverage: 1.0
# Hypothesis: EMA crosses capture momentum shifts. 4h EMA50 filter ensures trades align with higher timeframe trend.
# Volume confirmation filters low-conviction moves. Works in bull/bear via trend filter.
# Target: 15-37 trades/year via strict entry conditions.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_ema_cross_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h and 1d data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # 4h EMA(50) for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1d EMA(200) for long-term trend filter (avoid counter-trend in strong trends)
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # 1h EMA(20) and EMA(50) for entry signal
    ema_fast = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_slow = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(ema_200_1d_aligned[i]) or
            np.isnan(ema_fast[i]) or np.isnan(ema_slow[i])):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Trend filters
        uptrend_4h = close[i] > ema_50_4h_aligned[i]
        downtrend_4h = close[i] < ema_50_4h_aligned[i]
        uptrend_1d = close[i] > ema_200_1d_aligned[i]
        downtrend_1d = close[i] < ema_200_1d_aligned[i]
        
        # EMA cross signals
        golden_cross = ema_fast[i] > ema_slow[i] and ema_fast[i-1] <= ema_slow[i-1]
        death_cross = ema_fast[i] < ema_slow[i] and ema_fast[i-1] >= ema_slow[i-1]
        
        # Entry logic: EMA cross + volume + 4h trend alignment
        if (golden_cross and vol_confirm[i] and uptrend_4h and uptrend_1d and position != 1):
            position = 1
            signals[i] = 0.20
        elif (death_cross and vol_confirm[i] and downtrend_4h and downtrend_1d and position != -1):
            position = -1
            signals[i] = -0.20
        # Exit: EMA cross in opposite direction or 4h trend breakdown
        elif position == 1 and (death_cross or not uptrend_4h):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (golden_cross or not downtrend_4h):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals