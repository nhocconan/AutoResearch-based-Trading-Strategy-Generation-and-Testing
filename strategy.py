#!/usr/bin/env python3
"""
1h_TrendFollowing_With_4H1D_Filter_v1
Hypothesis: Combines 4h trend direction (EMA crossover) with 1d trend filter and volume confirmation on 1h timeframe.
Trades only during active session (08-20 UTC) to reduce noise. Uses EMA(13/34) on 4h for trend direction,
EMA(50) on 1d for long-term bias, and volume spike (>1.5x average) for entry confirmation on 1h.
Designed for 15-25 trades/year to minimize fee drift while capturing medium-term trends.
"""

name = "1h_TrendFollowing_With_4H1D_Filter_v1"
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
    
    # Pre-compute hour filter for session (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # 4h EMA crossover for trend direction (fast=13, slow=34)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_4h_fast = pd.Series(close_4h).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_4h_slow = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_4h_fast_aligned = align_htf_to_ltf(prices, df_4h, ema_4h_fast)
    ema_4h_slow_aligned = align_htf_to_ltf(prices, df_4h, ema_4h_slow)
    
    # 1d EMA for long-term trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):  # Start after 4h slow EMA warmup
        # Skip if any critical value is NaN
        if (np.isnan(ema_4h_fast_aligned[i]) or np.isnan(ema_4h_slow_aligned[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: only trade between 08:00-20:00 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if position == 0:
            # Long conditions: 4h fast EMA above slow EMA, price above 1d EMA, volume spike, in session
            if (ema_4h_fast_aligned[i] > ema_4h_slow_aligned[i] and 
                close[i] > ema_1d_aligned[i] and 
                volume[i] > 1.5 * vol_ma[i] and 
                in_session):
                signals[i] = 0.20
                position = 1
            # Short conditions: 4h fast EMA below slow EMA, price below 1d EMA, volume spike, in session
            elif (ema_4h_fast_aligned[i] < ema_4h_slow_aligned[i] and 
                  close[i] < ema_1d_aligned[i] and 
                  volume[i] > 1.5 * vol_ma[i] and 
                  in_session):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: 4h EMA crossover reverses OR price crosses below 1d EMA
            if (ema_4h_fast_aligned[i] < ema_4h_slow_aligned[i] or 
                close[i] < ema_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: 4h EMA crossover reverses OR price crosses above 1d EMA
            if (ema_4h_fast_aligned[i] > ema_4h_slow_aligned[i] or 
                close[i] > ema_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals