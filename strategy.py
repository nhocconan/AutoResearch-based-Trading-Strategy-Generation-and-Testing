#!/usr/bin/env python3
"""
4h_12h_KAMA_Slope_Change_With_Volume_and_Trend
Hypothesis: Detect trend changes via KAMA slope turning points on 4h timeframe, 
filtered by 12h trend direction and volume spike confirmation. 
KAMA adapts to market noise, reducing whipsaws in sideways markets. 
Only take longs when KAMA slope turns up (bullish acceleration) with 12h uptrend and volume spike, 
shorts when slope turns down (bearish acceleration) with 12h downtrend and volume spike.
Exit when KAMA slope reverses or volume drops. Designed for fewer, higher-quality trades.
"""

name = "4h_12h_KAMA_Slope_Change_With_Volume_and_Trend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # KAMA parameters
    er_len = 10
    fast_sc = 2
    slow_sc = 30
    
    # Efficiency Ratio
    change = np.abs(np.diff(close, prepend=close[0]))
    dir = np.abs(np.diff(close, prepend=close[0]))
    for i in range(1, len(change)):
        dir[i] = np.abs(close[i] - close[i-er_len]) if i >= er_len else 0
    er = np.where(dir != 0, change / dir, 0)
    er = np.where(er > 1, 1, er)
    
    # Smoothing constants
    sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
    
    # KAMA calculation
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # KAMA slope (first derivative)
    kama_slope = np.diff(kama, prepend=0)
    
    # Slope change detection: positive to negative or negative to positive
    # Bullish slope change: slope crosses above zero after being negative
    # Bearish slope change: slope crosses below zero after being positive
    bullish_slope_change = (kama_slope > 0) & (np.roll(kama_slope, 1) <= 0)
    bearish_slope_change = (kama_slope < 0) & (np.roll(kama_slope, 1) >= 0)
    # Avoid first element
    bullish_slope_change[0] = False
    bearish_slope_change[0] = False
    
    # Volume spike: >1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    # 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        if (np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(kama[i]) or
            np.isnan(kama_slope[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Bullish KAMA slope change + volume spike + price above 12h EMA50
            if (bullish_slope_change[i] and 
                volume_spike[i] and 
                close[i] > ema_50_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Bearish KAMA slope change + volume spike + price below 12h EMA50
            elif (bearish_slope_change[i] and 
                  volume_spike[i] and 
                  close[i] < ema_50_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bearish slope change OR price drops below 12h EMA50
            if bearish_slope_change[i] or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bullish slope change OR price rises above 12h EMA50
            if bullish_slope_change[i] or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals