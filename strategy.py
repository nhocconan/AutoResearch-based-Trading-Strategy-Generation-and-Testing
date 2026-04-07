#!/usr/bin/env python3
"""
1d_ma_crossover_1w_trend_filter_v1
Hypothesis: On daily timeframe, EMA(20) crosses EMA(50) with weekly trend filter and volume confirmation.
Long when EMA20 > EMA50 and weekly trend up, short when EMA20 < EMA50 and weekly trend down.
Uses volume spike to confirm breakout strength. Works in both bull and bear markets by following
the higher timeframe trend. Target: 10-20 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_ma_crossover_1w_trend_filter_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Daily EMAs for crossover signal
    ema_20 = pd.Series(close).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Weekly EMA for trend filter (20-period)
    ema_20w = df_1w['close'].ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # Align weekly EMA to daily timeframe
    ema_20w_aligned = align_htf_to_ltf(prices, df_1w, ema_20w)
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(ema_20[i]) or np.isnan(ema_50[i]) or 
            np.isnan(ema_20w_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter: weekly EMA20 > weekly EMA50 (approximate using price vs EMA20w)
        weekly_trend_up = close[i] > ema_20w_aligned[i]
        weekly_trend_down = close[i] < ema_20w_aligned[i]
        
        if position == 1:  # Long position
            # Exit: EMA20 crosses below EMA50 or weekly trend turns down
            if ema_20[i] < ema_50[i] or not weekly_trend_up:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: EMA20 crosses above EMA50 or weekly trend turns up
            if ema_20[i] > ema_50[i] or not weekly_trend_down:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: EMA20 crosses above EMA50 with volume and weekly trend up
            if (ema_20[i] > ema_50[i] and vol_confirm and weekly_trend_up):
                position = 1
                signals[i] = 0.25
            # Short entry: EMA20 crosses below EMA50 with volume and weekly trend down
            elif (ema_20[i] < ema_50[i] and vol_confirm and weekly_trend_down):
                position = -1
                signals[i] = -0.25
    
    return signals