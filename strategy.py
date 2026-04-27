#!/usr/bin/env python3
"""
Hypothesis: 6h timeframe with Elder Ray Index (Bull/Bear Power) + 1-day EMA50 trend filter + volume confirmation.
Elder Ray measures bull/bear power relative to EMA13, helping identify trend strength.
In bull markets: buy when bull power > 0 and rising, price above 1d EMA50, volume > 1.5x average.
In bear markets: sell when bear power < 0 and falling, price below 1d EMA50, volume > 1.5x average.
Exit when power crosses zero or reverses.
Designed to work in both bull (trend following) and bear (counter-trend on power reversal) regimes.
Target: 15-35 trades/year (~60-140 total over 4 years) to avoid fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema_period = 50
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= ema_period:
        ema_1d[ema_period - 1] = np.mean(close_1d[:ema_period])
        for i in range(ema_period, len(close_1d)):
            ema_1d[i] = (close_1d[i] * (2 / (ema_period + 1)) + 
                         ema_1d[i - 1] * (1 - (2 / (ema_period + 1))))
    
    # Calculate EMA13 for Elder Ray (6-period equivalent for 6h timeframe)
    ema13_period = 13
    ema13 = np.full(n, np.nan)
    if n >= ema13_period:
        ema13[ema13_period - 1] = np.mean(close[:ema13_period])
        for i in range(ema13_period, n):
            ema13[i] = (close[i] * (2 / (ema13_period + 1)) + 
                        ema13[i - 1] * (1 - (2 / (ema13_period + 1))))
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Smooth the power signals (3-period EMA) to reduce noise
    def smooth_ema(arr, period):
        smoothed = np.full_like(arr, np.nan)
        if len(arr) >= period and not np.isnan(arr[period-1]):
            smoothed[period-1] = np.nanmean(arr[:period])
            for i in range(period, len(arr)):
                if not np.isnan(arr[i]) and not np.isnan(smoothed[i-1]):
                    smoothed[i] = (arr[i] * (2 / (period + 1)) + 
                                   smoothed[i-1] * (1 - (2 / (period + 1))))
        return smoothed
    
    bull_power_smooth = smooth_ema(bull_power, 3)
    bear_power_smooth = smooth_ema(bear_power, 3)
    
    # Align 1d EMA50 to 6h timeframe
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume MA for confirmation (20-period)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i - 19:i + 1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need EMA13, smoothed power, EMA50, and volume MA20
    start_idx = max(ema13_period + 2, ema_period - 1, 19)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(bull_power_smooth[i]) or np.isnan(bear_power_smooth[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter: require volume above average
        vol_filter = vol_now > 1.5 * vol_avg
        
        if position == 0:
            # Long: Bull power positive AND rising, price above 1d EMA50, volume filter
            if (i > start_idx and 
                bull_power_smooth[i] > 0 and 
                bull_power_smooth[i] > bull_power_smooth[i-1] and
                price > ema_1d_aligned[i] and vol_filter):
                signals[i] = size
                position = 1
            # Short: Bear power negative AND falling, price below 1d EMA50, volume filter
            elif (i > start_idx and 
                  bear_power_smooth[i] < 0 and 
                  bear_power_smooth[i] < bear_power_smooth[i-1] and
                  price < ema_1d_aligned[i] and vol_filter):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Bull power crosses below zero OR starts falling
            if bull_power_smooth[i] <= 0 or (i > start_idx and bull_power_smooth[i] < bull_power_smooth[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: Bear power crosses above zero OR starts rising
            if bear_power_smooth[i] >= 0 or (i > start_idx and bear_power_smooth[i] > bear_power_smooth[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_ElderRay_Power_1dEMA50_Volume"
timeframe = "6h"
leverage = 1.0