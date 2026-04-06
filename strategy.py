#!/usr/bin/env python3
"""
6h Elder Ray Power with 1d Trend Filter and Volume Confirmation v1
Hypothesis: Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) measures
bull/bear strength relative to trend. Combined with 1d EMA50 trend filter and
volume confirmation, it captures strong directional moves while avoiding
counter-trend trades. Designed for 50-150 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_elder_ray_power_1d_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Elder Ray components on 6h
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13  # High - EMA13
    bear_power = ema13 - low   # EMA13 - Low
    
    # Volume filter
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(13, 20, 50)
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(ema13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(vol_ma[i]) or np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Determine trend from 1d EMA50
        uptrend = close[i] > ema50_1d_aligned[i]
        downtrend = close[i] < ema50_1d_aligned[i]
        
        # Check exits: Elder Ray divergence or trend change
        if position == 1:  # long position
            # Exit: Bear power turning positive (weakening) OR trend turns down
            if bear_power[i] > 0 or downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: Bull power turning negative (weakening) OR trend turns up
            if bull_power[i] < 0 or uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Strong Elder Ray + trend alignment + volume
            bull_entry = (bull_power[i] > 0 and  # Strong bull power
                         uptrend and           # Uptrend on 1d
                         volume[i] > vol_ma[i] * 1.5)  # Volume confirmation
            
            bear_entry = (bear_power[i] > 0 and  # Strong bear power
                         downtrend and          # Downtrend on 1d
                         volume[i] > vol_ma[i] * 1.5)  # Volume confirmation
            
            if bull_entry:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif bear_entry:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals