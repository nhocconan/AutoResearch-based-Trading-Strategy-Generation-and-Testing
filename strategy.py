#!/usr/bin/env python3
# 4h_12h_donchian_volume_breakout_v1
# Strategy: 4h Donchian breakout with 12h volume confirmation and trend filter
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Donchian channel breakouts capture strong trends. Volume confirms breakout strength. 
# 12h EMA200 filter ensures alignment with higher timeframe trend. 
# Long when price breaks above Donchian(20) high with volume > 1.5x 20-period average and price above 12h EMA200.
# Short when price breaks below Donchian(20) low with volume > 1.5x 20-period average and price below 12h EMA200.
# Exit on opposite Donchian breakout. Target 20-40 trades/year to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_donchian_volume_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for trend filter
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 200:
        return np.zeros(n)
    
    # 12h EMA(200) for trend filter
    close_12h = df_12h['close'].values
    ema_200_12h = pd.Series(close_12h).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_200_12h)
    
    # Donchian Channel (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_200_12h_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Donchian breakouts
        breakout_up = high[i] > donchian_high[i-1]
        breakout_down = low[i] < donchian_low[i-1]
        
        # Trend filter: price above/below 12h EMA200
        uptrend = close[i] > ema_200_12h_aligned[i]
        downtrend = close[i] < ema_200_12h_aligned[i]
        
        # Entry logic: Donchian breakout + volume + trend alignment
        if breakout_up and vol_confirm[i] and uptrend and position != 1:
            position = 1
            signals[i] = 0.25
        elif breakout_down and vol_confirm[i] and downtrend and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: opposite Donchian breakout
        elif position == 1 and breakout_down:
            position = 0
            signals[i] = 0.0
        elif position == -1 and breakout_up:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals