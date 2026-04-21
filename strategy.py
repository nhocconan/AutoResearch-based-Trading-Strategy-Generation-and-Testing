#!/usr/bin/env python3
"""
4h_1d_1w_Donchian_Breakout_Volume_Trend
Hypothesis: 4h Donchian(20) breakout with volume confirmation and 1w trend filter.
Long when price breaks above Donchian upper band with volume > 1.5x average and 1w close > 1w EMA50.
Short when price breaks below Donchian lower band with volume > 1.5x average and 1w close < 1w EMA50.
Exit when price crosses back through Donchian middle band (20-period average).
Designed for 4h timeframe to capture medium-term trends with ~20-40 trades/year.
Works in bull markets by buying breakouts and in bear markets by selling breakdowns.
Volume confirmation filters false breakouts.
Weekly trend filter ensures alignment with higher timeframe momentum.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load 1w data once for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # 1w EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Donchian channels on 4h data (using close prices)
    lookback = 20
    high_roll = prices['high'].rolling(window=lookback, min_periods=lookback).max().values
    low_roll = prices['low'].rolling(window=lookback, min_periods=lookback).min().values
    upper_band = high_roll
    lower_band = low_roll
    middle_band = (high_roll + low_roll) / 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if trend filter not ready
        if np.isnan(ema_50_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 1.5 * vol_ma
        else:
            volume_ok = False
        
        if position == 0:
            # Long conditions: break above upper band + volume + uptrend
            if price > upper_band[i] and volume_ok and price > ema_50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short conditions: break below lower band + volume + downtrend
            elif price < lower_band[i] and volume_ok and price < ema_50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses back below middle band
            if price < middle_band[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses back above middle band
            if price > middle_band[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1d_1w_Donchian_Breakout_Volume_Trend"
timeframe = "4h"
leverage = 1.0