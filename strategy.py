#!/usr/bin/env python3
"""
Hypothesis: 1h EMA crossover with 4h/1d trend filter and session timing.
- Long: EMA(9) > EMA(21) + 4h EMA(50) > EMA(100) + 1d close > 1d EMA(200) + session 08-20 UTC
- Short: EMA(9) < EMA(21) + 4h EMA(50) < EMA(100) + 1d close < 1d EMA(200) + session 08-20 UTC
- Exit: Opposite EMA(9)/EMA(21) cross
- Uses multiple timeframes for trend alignment (4h/1d for direction, 1h for entry timing)
- Session filter reduces noise during low-volume hours
- Target: 60-150 total trades over 4 years (15-37/year) on 1h timeframe
- Discrete position sizing: ±0.20 to minimize fee churn and manage drawdown
- Works in bull markets (trend alignment longs) and bear markets (trend alignment shorts)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # 1h EMAs for entry timing
    ema_9 = pd.Series(close).ewm(span=9, min_periods=9, adjust=False).mean().values
    ema_21 = pd.Series(close).ewm(span=21, min_periods=21, adjust=False).mean().values
    
    # 4h HTF for trend direction
    df_4h = get_htf_data(prices, '4h')
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_100_4h = pd.Series(df_4h['close'].values).ewm(span=100, min_periods=100, adjust=False).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    ema_100_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_100_4h)
    
    # 1d HTF for higher timeframe trend
    df_1d = get_htf_data(prices, '1d')
    ema_200_1d = pd.Series(df_1d['close'].values).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, df_1d['close'].values)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(open_time).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 200, 9, 21)  # Need 100 for 4h EMA100, 200 for 1d EMA200
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_9[i]) or 
            np.isnan(ema_21[i]) or
            np.isnan(ema_50_4h_aligned[i]) or
            np.isnan(ema_100_4h_aligned[i]) or
            np.isnan(ema_200_1d_aligned[i]) or
            np.isnan(close_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter
        in_session = session_filter[i]
        
        if position == 0:
            # Long: EMA(9) > EMA(21) + 4h EMA(50) > EMA(100) + 1d close > 1d EMA(200) + session
            if (ema_9[i] > ema_21[i] and 
                ema_50_4h_aligned[i] > ema_100_4h_aligned[i] and
                close_1d_aligned[i] > ema_200_1d_aligned[i] and
                in_session):
                signals[i] = 0.20
                position = 1
            # Short: EMA(9) < EMA(21) + 4h EMA(50) < EMA(100) + 1d close < 1d EMA(200) + session
            elif (ema_9[i] < ema_21[i] and 
                  ema_50_4h_aligned[i] < ema_100_4h_aligned[i] and
                  close_1d_aligned[i] < ema_200_1d_aligned[i] and
                  in_session):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: EMA(9) < EMA(21) (trend change)
            if ema_9[i] < ema_21[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: EMA(9) > EMA(21) (trend change)
            if ema_9[i] > ema_21[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_EMA_Crossover_4h1d_Trend_Session"
timeframe = "1h"
leverage = 1.0