#!/usr/bin/env python3
# 4h_12h_donchian_volume_trend_v1
# Strategy: 4h Donchian breakout with volume confirmation and 12h trend filter
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Donchian breakouts capture momentum. Volume confirms strength.
# 12h EMA50 filter prevents counter-trend trades. Low frequency (~20-40/year) to minimize fee drag.
# Works in bull (breakout long) and bear (breakdown short) with trend alignment.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_donchian_volume_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA(50) for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Donchian channels: 20-period high/low
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.8x 20-period average
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.8 * vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: price above/below 12h EMA50
        uptrend = close[i] > ema_50_12h_aligned[i]
        downtrend = close[i] < ema_50_12h_aligned[i]
        
        # Entry logic: Donchian breakout + volume + trend alignment
        if (close[i] > donchian_high[i] and vol_confirm[i] and uptrend and position != 1):
            position = 1
            signals[i] = 0.25
        elif (close[i] < donchian_low[i] and vol_confirm[i] and downtrend and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: trend reversal or opposite Donchian breakout
        elif position == 1 and (not uptrend or close[i] < donchian_low[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (not downtrend or close[i] > donchian_high[i]):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals