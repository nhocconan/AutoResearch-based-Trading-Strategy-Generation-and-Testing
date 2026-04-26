#!/usr/bin/env python3
"""
6h_EMA_Volume_Pullback_12hTrend_Filter_v1
Hypothesis: On 6h timeframe, buy pullbacks to 20-period EMA during 12h uptrend with volume confirmation, sell/short rallies to EMA during 12h downtrend. Uses discrete position sizing (0.0, ±0.25) to minimize fee churn. Designed to work in both bull (trend-following pullbacks) and bear (counter-trend bounces in range) markets by requiring 12h trend alignment, reducing false signals. Targets 50-150 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for HTF trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA34 for trend filter (stable, widely used)
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate 6h EMA20 for pullback entries
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume confirmation: volume > 1.5x 20-period EMA
    volume_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_confirm = volume > (volume_ema * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need sufficient data for all indicators)
    start_idx = max(100, 34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_12h_aligned[i]) or 
            np.isnan(ema_20[i]) or
            np.isnan(volume_ema[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # 12h trend filter
        uptrend = close[i] > ema_34_12h_aligned[i]
        downtrend = close[i] < ema_34_12h_aligned[i]
        
        # Long logic: price at or below EMA20 with volume confirmation in 12h uptrend
        if close[i] <= ema_20[i] and volume_confirm[i] and uptrend:
            if position != 1:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        # Short logic: price at or above EMA20 with volume confirmation in 12h downtrend
        elif close[i] >= ema_20[i] and volume_confirm[i] and downtrend:
            if position != -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = -0.25
        # Exit conditions: price moves back above/below EMA20 or trend weakens
        elif position == 1 and (close[i] > ema_20[i] or not uptrend):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close[i] < ema_20[i] or not downtrend):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_EMA_Volume_Pullback_12hTrend_Filter_v1"
timeframe = "6h"
leverage = 1.0