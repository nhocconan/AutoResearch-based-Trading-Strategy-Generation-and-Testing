#!/usr/bin/env python3
"""
4h_KAMA_Slope_12hTrend_VolumeFilter
Hypothesis: Uses KAMA (adaptive moving average) slope as primary trend signal, filtered by 12h EMA50 trend direction and volume spike (2x 48-bar average). KAMA adapts to market conditions, making it effective in both trending and ranging markets. Combined with 12h trend filter ensures we trade in higher timeframe direction, reducing whipsaws. Volume filter ensures momentum behind moves. Designed for low trade frequency (20-50/year) to minimize fee drag.
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
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate KAMA (Kaufman Adaptive Moving Average)
    # ER = Efficiency Ratio, SC = Smoothing Constant
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0)
    # Correct volatility calculation: sum of absolute changes over ER period
    volatility_sum = np.zeros_like(close)
    for i in range(1, len(close)):
        volatility_sum[i] = volatility_sum[i-1] + np.abs(close[i] - close[i-1])
        if i >= 10:
            volatility_sum[i] -= np.abs(close[i-10] - close[i-11]) if i >= 11 else 0
    
    er = np.zeros_like(close)
    er[9:] = np.abs(close[9:] - close[:-9]) / (volatility_sum[9:] + 1e-10)
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # for EMA(2)
    slow_sc = 2 / (30 + 1)  # for EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate KAMA slope (rate of change over 3 periods)
    kama_slope = np.zeros_like(close)
    kama_slope[3:] = (kama[3:] - kama[:-3]) / kama[:-3]
    
    # Volume confirmation: >2x 48-period MA (8 days of 4h bars)
    vol_ma_48 = pd.Series(volume).rolling(window=48, min_periods=48).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(kama_slope[i]) or
            np.isnan(vol_ma_48[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 12h EMA50
        uptrend = close[i] > ema_50_12h_aligned[i]
        downtrend = close[i] < ema_50_12h_aligned[i]
        
        # Volume confirmation (>2x average)
        vol_confirm = volume[i] > (2.0 * vol_ma_48[i])
        
        # Entry conditions: KAMA slope aligned with trend
        long_entry = kama_slope[i] > 0.001 and vol_confirm and uptrend
        short_entry = kama_slope[i] < -0.001 and vol_confirm and downtrend
        
        # Exit conditions: KAMA slope reverses or volume drops
        long_exit = kama_slope[i] < -0.0005
        short_exit = kama_slope[i] > 0.0005
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_KAMA_Slope_12hTrend_VolumeFilter"
timeframe = "4h"
leverage = 1.0