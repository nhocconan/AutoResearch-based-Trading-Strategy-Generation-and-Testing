#!/usr/bin/env python3
# 1h_4d_4h_1d_trend_v1
# Strategy: 1h breakout with 4h/1d trend filter, volume confirmation, and session filter (08-20 UTC)
# Timeframe: 1h
# Leverage: 1.0
# Hypothesis: In trending markets (identified by 4h/1d price > EMA50), buy breakouts above 1h rolling high with volume confirmation; sell/short breakdowns below 1h rolling low in downtrends. Uses 4h/1d for trend direction (low frequency), 1h for entry timing. Session filter reduces noise. Target: 15-37 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4d_4h_1d_trend_v1"
timeframe = "1h"
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
    
    # Pre-compute hour for session filter (08-20 UTC)
    hours = prices.index.hour  # index is DatetimeIndex
    
    # Lookback periods
    lookback = 20  # for 1h high/low breakout
    
    # 1h rolling high/low for breakout signals
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    roll_high = high_series.rolling(window=lookback, min_periods=lookback).max().values
    roll_low = low_series.rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_avg_20)
    
    # Load 4h data ONCE before loop for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(lookback, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(roll_high[i]) or np.isnan(roll_low[i])):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        # Trend filters: price above/below EMA50 on 4h and 1d
        uptrend_4h = close[i] > ema_50_4h_aligned[i]
        uptrend_1d = close[i] > ema_50_1d_aligned[i]
        downtrend_4h = close[i] < ema_50_4h_aligned[i]
        downtrend_1d = close[i] < ema_50_1d_aligned[i]
        
        # Breakout conditions
        breakout_up = high[i] > roll_high[i]  # new high
        breakdown_down = low[i] < roll_low[i]  # new low
        
        # Entry logic: breakout + volume + trend alignment + session
        if in_session and vol_confirm[i]:
            if breakout_up and uptrend_4h and uptrend_1d and position != 1:
                position = 1
                signals[i] = 0.20
            elif breakdown_down and downtrend_4h and downtrend_1d and position != -1:
                position = -1
                signals[i] = -0.20
        # Exit: opposite breakout with volume confirmation (regardless of session)
        elif position == 1 and breakdown_down and vol_confirm[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and breakout_up and vol_confirm[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals