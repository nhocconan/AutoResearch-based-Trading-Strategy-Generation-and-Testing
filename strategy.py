#!/usr/bin/env python3
# 12h_russell_bull_bear_power_v1
# Hypothesis: Combines daily Russell Bull/Bear Power with 12-hour price action to capture trend continuation in both bull and bear markets. Uses 1d EMA13 of (Close-EMA13) for bull power and (EMA13-Close) for bear power. Enters on 12h pullbacks to EMA21 with volume confirmation. Designed for low trade frequency (15-25/year) to avoid fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_russell_bull_bear_power_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate EMA13 on 12h for reference
    alpha13 = 2 / (13 + 1)
    ema13 = np.zeros(n)
    ema13[0] = close[0]
    for i in range(1, n):
        ema13[i] = alpha13 * close[i] + (1 - alpha13) * ema13[i-1]
    
    # Calculate EMA21 on 12h for pullback entries
    alpha21 = 2 / (21 + 1)
    ema21 = np.zeros(n)
    ema21[0] = close[0]
    for i in range(1, n):
        ema21[i] = alpha21 * close[i] + (1 - alpha21) * ema21[i-1]
    
    # Get daily data for Russell Bull/Bear Power
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate EMA13 on daily
    alpha13d = 2 / (13 + 1)
    ema13_1d = np.zeros(len(df_1d))
    ema13_1d[0] = close_1d[0]
    for i in range(1, len(df_1d)):
        ema13_1d[i] = alpha13d * close_1d[i] + (1 - alpha13d) * ema13_1d[i-1]
    
    # Russell Bull Power = Close - EMA13
    bull_power = close_1d - ema13_1d
    # Russell Bear Power = EMA13 - Close
    bear_power = ema13_1d - close_1d
    
    # Smooth with EMA13
    def ema(arr, alpha):
        res = np.zeros_like(arr)
        res[0] = arr[0]
        for i in range(1, len(arr)):
            res[i] = alpha * arr[i] + (1 - alpha) * res[i-1]
        return res
    
    bull_power_smooth = ema(bull_power, alpha13d)
    bear_power_smooth = ema(bear_power, alpha13d)
    
    # Combine: positive = bullish, negative = bearish
    russell_osc = bull_power_smooth - bear_power_smooth  # equivalent to 2*(Close-EMA13)
    
    # Align to 12h
    russell_osc_12h = align_htf_to_ltf(prices, df_1d, russell_osc)
    
    # Volume confirmation: 20-period average
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
        if np.isnan(ema21[i]) or np.isnan(russell_osc_12h[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_ok = volume[i] > vol_ma_20[i] * 1.5
        
        if position == 1:  # Long position
            # Exit: Russell turns bearish or price breaks below EMA21
            if russell_osc_12h[i] < 0 or close[i] < ema21[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Russell turns bullish or price breaks above EMA21
            if russell_osc_12h[i] > 0 or close[i] > ema21[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: bullish Russell + pullback to EMA21 + volume
            if (russell_osc_12h[i] > 0 and 
                low[i] <= ema21[i] * 1.005 and  # Allow small tolerance
                close[i] > ema21[i] and
                vol_ok):
                position = 1
                signals[i] = 0.25
            # Enter short: bearish Russell + pullback to EMA21 + volume
            elif (russell_osc_12h[i] < 0 and 
                  high[i] >= ema21[i] * 0.995 and  # Allow small tolerance
                  close[i] < ema21[i] and
                  vol_ok):
                position = -1
                signals[i] = -0.25
    
    return signals