#!/usr/bin/env python3
# 4h_1d_elder_ray_volume_v1
# Strategy: 4h Elder Ray Index with volume confirmation and 1d trend filter
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Elder Ray measures bull/bear power via EMA13. Bull Power > 0 with rising Bear Power and volume confirmation indicates strength in uptrends; Bear Power < 0 with falling Bull Power and volume indicates weakness in downtrends. Uses 1d EMA50 for trend filter to avoid counter-trend trades. Targets 20-40 trades/year to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_elder_ray_volume_v1"
timeframe = "4h"
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 4h EMA(13) for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # Bull Power: high minus EMA13
    bear_power = low - ema_13   # Bear Power: low minus EMA13
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: price above/below 1d EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Entry logic: Elder Ray + volume + trend alignment
        if (bull_power[i] > 0 and bear_power[i] > bear_power[i-1] and  # Bull power positive and rising
            vol_confirm[i] and uptrend and position != 1):
            position = 1
            signals[i] = 0.25
        elif (bear_power[i] < 0 and bull_power[i] < bull_power[i-1] and  # Bear power negative and falling
              vol_confirm[i] and downtrend and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: Elder Ray divergence or trend change
        elif position == 1 and (bull_power[i] <= 0 or not uptrend):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (bear_power[i] >= 0 or not downtrend):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals