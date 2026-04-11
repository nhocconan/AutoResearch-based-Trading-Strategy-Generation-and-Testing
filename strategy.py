#!/usr/bin/env python3
# 6h_1d_elder_ray_volume_v1
# Strategy: 6h Elder Ray Index (Bull/Bear Power) with volume confirmation and 1d trend filter
# Timeframe: 6h
# Leverage: 1.0
# Hypothesis: Elder Ray measures bull/bear power relative to EMA13. In bull markets, Bull Power > 0 with rising trend and volume confirmation. In bear markets, Bear Power < 0 with falling trend and volume confirmation. Uses 1d EMA50 for trend filter to avoid counter-trend trades. Low frequency (~20-40/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_elder_ray_volume_v1"
timeframe = "6h"
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Elder Ray calculation: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean()
    bull_power = high - ema13.values
    bear_power = low - ema13.values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(13, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: price above/below 1d EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Entry logic: Elder Ray + volume + trend alignment
        if (bull_power[i] > 0 and bull_power[i] > bull_power[i-1] and  # Bull Power positive and rising
            vol_confirm[i] and uptrend and position != 1):
            position = 1
            signals[i] = 0.25
        elif (bear_power[i] < 0 and bear_power[i] < bear_power[i-1] and  # Bear Power negative and falling
              vol_confirm[i] and downtrend and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: Elder Ray momentum divergence or trend change
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