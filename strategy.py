#!/usr/bin/env python3
# 12h_1w_donchian_volume_trend_v1
# Strategy: 12h Donchian(20) breakout with 1w trend filter and volume confirmation
# Timeframe: 12h
# Leverage: 1.0
# Hypothesis: Donchian breakouts capture momentum; 1w EMA50 filter ensures alignment with higher-timeframe trend; volume confirms breakout strength. Works in bull via upward breakouts, in bear via downward breakouts. Low trade frequency (~15-30/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_donchian_volume_trend_v1"
timeframe = "12h"
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
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 12h Donchian(20)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_confirm[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: price above/below 1w EMA50
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        # Entry logic: Donchian breakout + trend alignment + volume confirmation
        if close[i] > donch_high[i] and uptrend and vol_confirm[i] and position != 1:
            position = 1
            signals[i] = 0.25
        elif close[i] < donch_low[i] and downtrend and vol_confirm[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: opposite Donchian breakout
        elif position == 1 and close[i] < donch_low[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > donch_high[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals