#!/usr/bin/env python3
# 6h_1w_1d_elder_ray_volume_v1
# Strategy: Elder Ray (Bull/Bear Power) + volume confirmation + 1w trend filter
# Timeframe: 6h
# Leverage: 1.0
# Hypothesis: Elder Ray measures bull/bear power via EMA13. In bull markets, bull power > 0 with volume confirms strength. In bear markets, bear power < 0 with volume confirms weakness. 1w EMA40 filters for major trend. Low trade frequency to avoid fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_1d_elder_ray_volume_v1"
timeframe = "6h"
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
    
    # Load 1d data ONCE before loop for EMA13 (Elder Ray)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1d EMA13 for Elder Ray
    close_1d = df_1d['close'].values
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema13_1d)
    
    # 1w EMA40 for trend filter
    close_1w = df_1w['close'].values
    ema40_1w = pd.Series(close_1w).ewm(span=40, adjust=False, min_periods=40).mean().values
    ema40_1w_aligned = align_htf_to_ltf(prices, df_1w, ema40_1w)
    
    # Elder Ray components
    bull_power = high - ema13_1d_aligned
    bear_power = low - ema13_1d_aligned
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(ema13_1d_aligned[i]) or np.isnan(ema40_1w_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: price above/below 1w EMA40
        uptrend = close[i] > ema40_1w_aligned[i]
        downtrend = close[i] < ema40_1w_aligned[i]
        
        # Entry logic: Elder Ray + volume + trend alignment
        if bull_power[i] > 0 and vol_confirm[i] and uptrend and position != 1:
            position = 1
            signals[i] = 0.25
        elif bear_power[i] < 0 and vol_confirm[i] and downtrend and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: power crosses zero
        elif position == 1 and bull_power[i] <= 0:
            position = 0
            signals[i] = 0.0
        elif position == -1 and bear_power[i] >= 0:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals