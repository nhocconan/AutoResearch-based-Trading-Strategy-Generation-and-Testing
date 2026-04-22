#!/usr/bin/env python3
"""
Hypothesis: 12-hour Williams Alligator with 1-week trend filter and volume confirmation.
Long when Alligator jaws (13-period smoothed median) crosses above teeth (8-period smoothed median) with
price above lips (5-period smoothed median) and volume above 20-period average.
Short when jaws cross below teeth with price below lips and volume above 20-period average.
Exit when jaws re-cross teeth in opposite direction.
Alligator uses SMMA (smoothed moving average) which lags less than EMA/SMA and whipsaws less in ranging markets.
Weekly trend filter ensures alignment with higher timeframe momentum.
Designed for low trade frequency (~10-25/year) to avoid fee drag while capturing sustained trends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(arr, period):
    """Smoothed Moving Average (SMMA) - also called Wilder's smoothing"""
    if len(arr) < period:
        return np.full_like(arr, np.nan, dtype=float)
    result = np.full_like(arr, np.nan, dtype=float)
    # First value is simple average
    result[period-1] = np.mean(arr[:period])
    # Subsequent values: (prev*(period-1) + current) / period
    for i in range(period, len(arr)):
        result[i] = (result[i-1] * (period-1) + arr[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    median = (high + low) / 2.0  # Alligator uses median price
    
    # Load 1-week data for trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly trend: 50-period EMA of weekly close
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_uptrend = close_1w > ema_50_1w
    weekly_downtrend = close_1w < ema_50_1w
    
    # Align weekly trend to 12h timeframe (wait for weekly close)
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend.astype(float))
    weekly_downtrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_downtrend.astype(float))
    
    # Alligator components on 12h timeframe
    jaws = smma(median, 13)  # Blue line
    teeth = smma(median, 8)  # Red line
    lips = smma(median, 5)   # Green line
    
    # Volume filter: 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(40, n):
        # Skip if data not ready
        if (np.isnan(jaws[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(weekly_uptrend_aligned[i]) or 
            np.isnan(weekly_downtrend_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Jaws cross above teeth, price above lips, weekly uptrend, volume confirmation
            if (jaws[i] > teeth[i] and jaws[i-1] <= teeth[i-1] and  # crossover
                close[i] > lips[i] and
                weekly_uptrend_aligned[i] > 0.5 and
                volume[i] > 1.5 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: Jaws cross below teeth, price below lips, weekly downtrend, volume confirmation
            elif (jaws[i] < teeth[i] and jaws[i-1] >= teeth[i-1] and  # crossover
                  close[i] < lips[i] and
                  weekly_downtrend_aligned[i] > 0.5 and
                  volume[i] > 1.5 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Jaws cross back below teeth
                if jaws[i] < teeth[i] and jaws[i-1] >= teeth[i-1]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Jaws cross back above teeth
                if jaws[i] > teeth[i] and jaws[i-1] <= teeth[i-1]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_WilliamsAlligator_1wTrend_Filter_Volume"
timeframe = "12h"
leverage = 1.0