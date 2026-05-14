#!/usr/bin/env python3
# 4h_russell_bull_bear_power_v1
# Hypothesis: Combines Russell 2000 Bull/Bear Power with 4-hour trend following. 
# Bull Power = High - EMA13, Bear Power = EMA13 - Low. 
# Long when Bull Power > 0 and rising, short when Bear Power > 0 and falling.
# Uses 1-day trend filter to avoid counter-trend trades. Target: 20-40 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_russell_bull_bear_power_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate EMA13 for Bull/Bear Power
    alpha = 2 / (13 + 1)
    ema13 = np.zeros(n)
    ema13[0] = close[0]
    for i in range(1, n):
        ema13[i] = alpha * close[i] + (1 - alpha) * ema13[i-1]
    
    # Calculate Bull Power and Bear Power
    bull_power = high - ema13
    bear_power = ema13 - low
    
    # Calculate 1-day trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate EMA50 on daily
    alpha_1d = 2 / (50 + 1)
    ema50_1d = np.zeros(len(df_1d))
    ema50_1d[0] = close_1d[0]
    for i in range(1, len(df_1d)):
        ema50_1d[i] = alpha_1d * close_1d[i] + (1 - alpha_1d) * ema50_1d[i-1]
    
    # Daily trend: 1 if close > EMA50, -1 if close < EMA50
    trend_1d = np.where(close_1d > ema50_1d, 1, -1)
    trend_4h = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # Volume filter: 20-period average
    vol_ma_20 = np.zeros(n)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma_20[i] = vol_sum / 20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if np.isnan(ema13[i]) or np.isnan(trend_4h[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_ok = volume[i] > vol_ma_20[i] * 1.5
        
        if position == 1:  # Long position
            # Exit: Bear Power > 0 and increasing OR trend turns bearish
            if bear_power[i] > 0 and bear_power[i] > bear_power[i-1] or trend_4h[i] == -1:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Bull Power > 0 and increasing OR trend turns bullish
            if bull_power[i] > 0 and bull_power[i] > bull_power[i-1] or trend_4h[i] == 1:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: Bull Power > 0 and rising, bullish trend, volume
            if (bull_power[i] > 0 and 
                bull_power[i] > bull_power[i-1] and 
                trend_4h[i] == 1 and 
                vol_ok):
                position = 1
                signals[i] = 0.25
            # Enter short: Bear Power > 0 and rising, bearish trend, volume
            elif (bear_power[i] > 0 and 
                  bear_power[i] > bear_power[i-1] and 
                  trend_4h[i] == -1 and 
                  vol_ok):
                position = -1
                signals[i] = -0.25
    
    return signals