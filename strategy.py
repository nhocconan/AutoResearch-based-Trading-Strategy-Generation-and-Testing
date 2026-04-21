#!/usr/bin/env python3
"""
12h_1d_1w_Chaikin_Money_Flow_Trend_Follow_v1
Hypothesis: Follow weekly trend using EMA34, enter on pullbacks when Chaikin Money Flow confirms institutional accumulation/distribution.
Long when weekly EMA34 up, price pulls back to EMA20, and CMF > 0.15.
Short when weekly EMA34 down, price pulls back to EMA20, and CMF < -0.15.
Exit when CMF crosses zero or price moves against weekly trend.
Designed for low trade frequency (~15-30 trades/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtd_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data for CMF calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Chaikin Money Flow (20-period)
    # Money Flow Multiplier = [(Close - Low) - (High - Close)] / (High - Low)
    mf_multiplier = np.where((high_1d - low_1d) != 0, 
                             ((close_1d - low_1d) - (high_1d - close_1d)) / (high_1d - low_1d), 
                             0)
    money_flow_volume = mf_multiplier * volume_1d
    
    # CMF = 20-period sum of Money Flow Volume / 20-period sum of Volume
    mfv_sum = pd.Series(money_flow_volume).rolling(window=20, min_periods=20).sum().values
    vol_sum = pd.Series(volume_1d).rolling(window=20, min_periods=20).sum().values
    cmf = np.where(vol_sum != 0, mfv_sum / vol_sum, 0)
    
    # Align CMF to 12h timeframe
    cmf_aligned = align_htf_to_ltf(prices, df_1d, cmf)
    
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate EMA34 on weekly
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    # Align to 12h timeframe
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate EMA20 on 12h for pullback entries
    close = prices['close'].values
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(cmf_aligned[i]) or np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        
        # Weekly trend filter: EMA34 slope
        if i >= 51:
            ema34_prev = ema34_1w_aligned[i-1]
            ema34_curr = ema34_1w_aligned[i]
            weekly_uptrend = ema34_curr > ema34_prev
            weekly_downtrend = ema34_curr < ema34_prev
        else:
            weekly_uptrend = False
            weekly_downtrend = False
        
        if position == 0:
            # Long conditions: weekly uptrend + price near EMA20 (pullback) + CMF accumulation
            if weekly_uptrend and price <= ema20[i] * 1.01 and price >= ema20[i] * 0.99 and cmf_aligned[i] > 0.15:
                signals[i] = 0.25
                position = 1
            # Short conditions: weekly downtrend + price near EMA20 (pullback) + CMF distribution
            elif weekly_downtrend and price >= ema20[i] * 0.99 and price <= ema20[i] * 1.01 and cmf_aligned[i] < -0.15:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: CMF turns negative or weekly trend breaks down
            if cmf_aligned[i] < 0 or not weekly_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: CMF turns positive or weekly trend breaks up
            if cmf_aligned[i] > 0 or not weekly_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1d_1w_Chaikin_Money_Flow_Trend_Follow_v1"
timeframe = "12h"
leverage = 1.0