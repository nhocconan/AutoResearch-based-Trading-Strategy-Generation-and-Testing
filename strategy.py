#!/usr/bin/env python3
# 1d_1w_donchian_breakout_volume_v1
# Strategy: Daily Donchian(20) breakout with volume confirmation and weekly trend filter
# Timeframe: 1d
# Leverage: 1.0
# Hypothesis: Buy when price breaks above 20-day high with above-average volume in a weekly uptrend.
# Sell/short when price breaks below 20-day low with above-average volume in a weekly downtrend.
# Weekly trend filter avoids counter-trend trades. Low frequency (~10-25/year) minimizes fee drag.
# Works in both bull and bear markets by only trading with the weekly trend.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_donchian_breakout_volume_v1"
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
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Daily Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-day average
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_avg_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Weekly trend filter
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        # Entry logic: Donchian breakout + volume + weekly trend alignment
        if (high[i] > donchian_high[i-1] and vol_confirm[i] and uptrend and position != 1):
            position = 1
            signals[i] = 0.25
        elif (low[i] < donchian_low[i-1] and vol_confirm[i] and downtrend and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: opposite breakout or trend change
        elif position == 1 and (low[i] < donchian_low[i-1] or not uptrend):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (high[i] > donchian_high[i-1] or not downtrend):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals