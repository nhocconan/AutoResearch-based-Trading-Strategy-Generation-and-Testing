#!/usr/bin/env python3
"""
1d_KAMA_Trend_Filter_WeeklyTrend_v2
Hypothesis: KAMA trend following on 1d with weekly trend filter and volume confirmation.
Uses KAMA to filter whipsaws and weekly trend to avoid counter-trend trades.
Volume spike (>2x 20-bar MA) confirms momentum. Designed for low trade frequency
(7-25 trades/year) to minimize fee drag while capturing strong trends.
Works in both bull and bear markets by following weekly trend direction.
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
    
    # Get 1w data for weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate KAMA on 1d (approximated via close prices)
    # Efficiency ratio: |close - close[10]| / sum(|close - close[1]|) over 10 periods
    change = np.abs(np.subtract(close[10:], close[:-10]))
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0)  # Will adjust below
    
    # Proper ER calculation
    er = np.zeros_like(close)
    for i in range(10, len(close)):
        if i >= 10:
            price_change = np.abs(close[i] - close[i-10])
            volatility_sum = np.sum(np.abs(np.diff(close[i-9:i+1])))
            if volatility_sum > 0:
                er[i] = price_change / volatility_sum
            else:
                er[i] = 0
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Volume confirmation: >2x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(kama[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Weekly trend filter
        weekly_uptrend = close[i] > ema_34_1w_aligned[i]
        weekly_downtrend = close[i] < ema_34_1w_aligned[i]
        
        # KAMA trend
        kama_uptrend = close[i] > kama[i]
        kama_downtrend = close[i] < kama[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > (2.0 * vol_ma_20[i])
        
        # Entry: KAMA aligned with weekly trend + volume
        long_entry = vol_confirm and weekly_uptrend and kama_uptrend
        short_entry = vol_confirm and weekly_downtrend and kama_downtrend
        
        # Exit: opposite KAMA signal or weekly trend change
        long_exit = kama_downtrend or (not weekly_uptrend)
        short_exit = kama_uptrend or (not weekly_downtrend)
        
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

name = "1d_KAMA_Trend_Filter_WeeklyTrend_v2"
timeframe = "1d"
leverage = 1.0