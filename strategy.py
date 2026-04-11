#!/usr/bin/env python3
# 1h_4h_1d_trend_ema_volume_v1
# Strategy: 1h EMA crossover with 4h EMA trend filter, 1d EMA200 regime filter, and volume confirmation
# Timeframe: 1h
# Leverage: 1.0
# Hypothesis: In trending markets (4h EMA50 aligned with price), 1h EMA(9/21) crossovers capture momentum.
# Volume confirms strength of breakout. 1d EMA200 avoids counter-trend trades in strong regimes.
# Designed for low frequency (15-35 trades/year) to minimize fee drag in 1h timeframe.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_trend_ema_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
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
    
    # 1d EMA(200) for regime filter
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # 1h EMA(9) and EMA(21) for entry signals
    ema_9 = pd.Series(close).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema_21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_avg_20)
    
    # Session filter: 08-20 UTC (reduces noise outside active hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(ema_9[i]) or np.isnan(ema_21[i])):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Skip if outside trading session
        if not session_filter[i]:
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Determine trend and regime
        uptrend_4h = close[i] > ema_50_4h_aligned[i]
        downtrend_4h = close[i] < ema_50_4h_aligned[i]
        bull_regime = close[i] > ema_200_1d_aligned[i]
        bear_regime = close[i] < ema_200_1d_aligned[i]
        
        # EMA crossover signals
        ema_cross_up = ema_9[i] > ema_21[i] and ema_9[i-1] <= ema_21[i-1]
        ema_cross_down = ema_9[i] < ema_21[i] and ema_9[i-1] >= ema_21[i-1]
        
        # Entry logic: EMA crossover + volume + trend/regime alignment
        if ema_cross_up and vol_confirm[i] and uptrend_4h and bull_regime and position != 1:
            position = 1
            signals[i] = 0.20
        elif ema_cross_down and vol_confirm[i] and downtrend_4h and bear_regime and position != -1:
            position = -1
            signals[i] = -0.20
        # Exit: EMA crossover in opposite direction or regime/trend change
        elif position == 1 and (ema_cross_down or not bull_regime or not uptrend_4h):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (ema_cross_up or not bear_regime or not downtrend_4h):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals