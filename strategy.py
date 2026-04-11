#!/usr/bin/env python3
# 12h_1d_donchian_breakout_volume_v1
# Strategy: 12h Donchian breakout with volume confirmation and 1d trend filter
# Timeframe: 12h
# Leverage: 1.0
# Hypothesis: Donchian breakouts capture trends with clear entry/exit rules. Volume confirmation filters false breakouts. 1d EMA50 ensures trades align with higher timeframe trend. Designed for low trade frequency (15-30/year) to minimize fee drag while capturing major trends in both bull and bear markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_donchian_breakout_volume_v1"
timeframe = "12h"
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
    
    # 12h Donchian(20) channels
    donchian_window = 20
    high_max = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    low_min = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(donchian_window, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(high_max[i]) or 
            np.isnan(low_min[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: price above/below 1d EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Entry logic: Donchian breakout + volume + trend alignment
        if (high[i] > high_max[i-1] and vol_confirm[i] and uptrend and position != 1):
            position = 1
            signals[i] = 0.25
        elif (low[i] < low_min[i-1] and vol_confirm[i] and downtrend and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: trend reversal or opposite breakout
        elif position == 1 and (not uptrend or low[i] < low_min[i-1]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (not downtrend or high[i] > high_max[i-1]):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals