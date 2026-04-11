#!/usr/bin/env python3
# 4h_1d_donchian_breakout_volume_trend_v1
# Strategy: 4h Donchian breakout with 1d EMA trend filter and volume confirmation
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Donchian breakouts capture strong momentum. Combined with 1d EMA trend filter and volume confirmation, this reduces false breakouts and works in both bull and bear markets by following the higher timeframe trend. Designed for low trade frequency (~20-40/year) to avoid fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_donchian_breakout_volume_trend_v1"
timeframe = "4h"
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA100 for trend filter
    ema_100_1d = pd.Series(df_1d['close'].values).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema_100_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_100_1d)
    
    # 20-period Donchian channels (using previous period to avoid look-ahead)
    donchian_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # 20-period volume average for confirmation
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_100_1d_aligned[i]) or 
            np.isnan(donchian_high_20[i]) or np.isnan(donchian_low_20[i]) or
            np.isnan(vol_avg_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_avg_20[i]
        
        # Breakout signals using Donchian levels (from previous period)
        breakout_up = high[i] > donchian_high_20[i]
        breakdown_down = low[i] < donchian_low_20[i]
        
        # 1d EMA trend filter: price above EMA = bullish trend, below = bearish
        trend_bullish = close[i] > ema_100_1d_aligned[i]
        trend_bearish = close[i] < ema_100_1d_aligned[i]
        
        # Entry conditions
        # Long: Breakout above Donchian high AND bullish trend AND volume confirmation
        if breakout_up and trend_bullish and vol_confirm and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: Breakdown below Donchian low AND bearish trend AND volume confirmation
        elif breakdown_down and trend_bearish and vol_confirm and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: Opposite breakout using Donchian levels
        elif position == 1 and low[i] < donchian_low_20[i]:  # Break below Donchian low
            position = 0
            signals[i] = 0.0
        elif position == -1 and high[i] > donchian_high_20[i]:  # Break above Donchian high
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals