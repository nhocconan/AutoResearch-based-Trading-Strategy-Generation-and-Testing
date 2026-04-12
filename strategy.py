#!/usr/bin/env python3
"""
1h_4d_Trend_Follow_With_Volume_Filter
Hypothesis: Use 4-hour EMA trend (EMA20) as primary direction, 1-hour breakout of 20-bar high/low with volume confirmation (>1.5x 20-bar average volume) for entry timing. Trade only during 08-20 UTC to avoid low-liquidity periods. Fixed position size 0.20 to minimize churn. Designed to capture trends in both bull and bear markets while avoiding whipsaws via volume confirmation and time filtering.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4d_Trend_Follow_With_Volume_Filter"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4H EMA TREND (HTF) ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # === 1H INDICATORS ===
    # 20-period high/low for breakout
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if not ready
        if (np.isnan(ema_20_4h_aligned[i]) or np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # Trend direction from 4h EMA20
        trend_up = close[i] > ema_20_4h_aligned[i]
        trend_down = close[i] < ema_20_4h_aligned[i]
        
        # Breakout conditions
        # Long: price breaks above 20-period high with volume + uptrend
        long_breakout = (close[i] > high_20[i]) and vol_confirmed and trend_up
        
        # Short: price breaks below 20-period low with volume + downtrend
        short_breakout = (close[i] < low_20[i]) and vol_confirmed and trend_down
        
        # Exit: reverse signal from trend
        exit_long = (position == 1) and (not trend_up)
        exit_short = (position == -1) and (not trend_down)
        
        # Execute trades
        if long_breakout and position != 1:
            position = 1
            signals[i] = 0.20
        elif short_breakout and position != -1:
            position = -1
            signals[i] = -0.20
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals