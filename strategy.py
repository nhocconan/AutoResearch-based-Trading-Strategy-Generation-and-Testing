#!/usr/bin/env python3
# 1h_4d_donchian_breakout_v1
# Strategy: 1h Donchian breakout with 4h trend filter and volume confirmation
# Timeframe: 1h
# Leverage: 1.0
# Hypothesis: Donchian channel breakouts capture momentum. 4h EMA trend filter ensures trades align with higher timeframe direction, reducing false breakouts in sideways markets. Volume confirmation adds validity. Designed for low trade frequency (~20-40/year) to avoid fee drag. Works in both bull and bear markets by following the 4h trend.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4d_donchian_breakout_v1"
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
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period) from 1h data
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 4h EMA50 for trend filter
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 20-period volume average for confirmation
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_avg_20[i])):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_avg_20[i]
        
        # Breakout signals using Donchian levels
        breakout_up = high[i] > donchian_high[i-1]
        breakdown_down = low[i] < donchian_low[i-1]
        
        # 4h EMA trend filter: price above EMA = bullish trend, below = bearish
        trend_bullish = close[i] > ema_50_4h_aligned[i]
        trend_bearish = close[i] < ema_50_4h_aligned[i]
        
        # Entry conditions
        # Long: Breakout above Donchian high AND bullish trend AND volume confirmation
        if breakout_up and trend_bullish and vol_confirm and position != 1:
            position = 1
            signals[i] = 0.20
        # Short: Breakdown below Donchian low AND bearish trend AND volume confirmation
        elif breakdown_down and trend_bearish and vol_confirm and position != -1:
            position = -1
            signals[i] = -0.20
        # Exit: Opposite breakout using Donchian levels
        elif position == 1 and low[i] < donchian_low[i-1]:  # Break below Donchian low
            position = 0
            signals[i] = 0.0
        elif position == -1 and high[i] > donchian_high[i-1]:  # Break above Donchian high
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals