#!/usr/bin/env python3
"""
4h_1d_1w_Donchian20_Breakout_Volume_Regime_Filtered_v1
Hypothesis: Donchian(20) breakout on 4h with volume confirmation and weekly trend filter.
Long when price > 20-bar high + volume > 1.5x 20-bar average + weekly EMA34 rising.
Short when price < 20-bar low + volume > 1.5x 20-bar average + weekly EMA34 falling.
Exit when price crosses 20-bar mid-point (mean of high/low).
Uses 4h as primary timeframe for signal generation, with 1w for trend filter.
Target: 20-40 trades/year per symbol. Works in bull/bear by following weekly trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load weekly data once for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 35:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate EMA34 on weekly
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    # Align to 4h timeframe
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate Donchian channels on 4h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # 20-period high and low
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    # Mid-point for exit
    mid_20 = (high_20 + low_20) / 2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if indicators not ready
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(mid_20[i]) or np.isnan(ema34_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        volume = prices['volume'].iloc[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 1.5 * vol_ma
        else:
            volume_ok = False
        
        # Weekly trend filter: EMA34 slope
        if i >= 61:
            ema34_prev = ema34_1w_aligned[i-1]
            ema34_curr = ema34_1w_aligned[i]
            weekly_uptrend = ema34_curr > ema34_prev
            weekly_downtrend = ema34_curr < ema34_prev
        else:
            weekly_uptrend = False
            weekly_downtrend = False
        
        if position == 0:
            # Long conditions: break above 20-bar high + volume + weekly uptrend
            if price > high_20[i] and volume_ok and weekly_uptrend:
                signals[i] = 0.25
                position = 1
            # Short conditions: break below 20-bar low + volume + weekly downtrend
            elif price < low_20[i] and volume_ok and weekly_downtrend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses back below 20-bar mid-point
            if price < mid_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses back above 20-bar mid-point
            if price > mid_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1d_1w_Donchian20_Breakout_Volume_Regime_Filtered_v1"
timeframe = "4h"
leverage = 1.0