#/usr/bin/env python3
"""
1D EMA Crossover with Weekly Momentum Filter
Hypothesis: On daily timeframe, EMA crossover (21/55) signals trend changes.
Weekly EMA (89) acts as a higher-timeframe filter to avoid counter-trend trades.
Only take trades when daily EMA crossover aligns with weekly trend.
This reduces whipsaws in ranging markets while capturing strong trends in both bull and bear markets.
Low trade frequency expected (<25/year) due to strict alignment requirement.
"""

name = "1d_EMA21_55_Crossover_WeeklyFilter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 90:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA89 for trend filter
    ema89_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 89:
        ema89_1w[88] = np.mean(close_1w[0:89])
        for i in range(89, len(close_1w)):
            ema89_1w[i] = (close_1w[i] * 2 + ema89_1w[i-1] * 87) / 89
    
    # Align weekly EMA89 to daily timeframe
    ema89_1w_aligned = align_htf_to_ltf(prices, df_1w, ema89_1w)
    
    # Calculate daily EMA21 and EMA55
    ema21 = np.full_like(close, np.nan)
    ema55 = np.full_like(close, np.nan)
    
    if len(close) >= 55:
        # Initialize EMA21
        ema21[20] = np.mean(close[0:21])
        for i in range(21, len(close)):
            ema21[i] = (close[i] * 2 + ema21[i-1] * 19) / 21
        
        # Initialize EMA55
        ema55[54] = np.mean(close[0:55])
        for i in range(55, len(close)):
            ema55[i] = (close[i] * 2 + ema55[i-1] * 53) / 55
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 55  # Need EMA55 ready
    
    for i in range(start_idx, n):
        # Skip if weekly trend data not ready
        if np.isnan(ema89_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if daily EMAs not ready
        if np.isnan(ema21[i]) or np.isnan(ema55[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine signals
        daily_bullish = ema21[i] > ema55[i]
        daily_bearish = ema21[i] < ema55[i]
        weekly_uptrend = close[i] > ema89_1w_aligned[i]
        weekly_downtrend = close[i] < ema89_1w_aligned[i]
        
        if position == 0:
            # Enter long: daily bullish crossover AND weekly uptrend
            if daily_bullish and weekly_uptrend:
                signals[i] = 0.25
                position = 1
            # Enter short: daily bearish crossover AND weekly downtrend
            elif daily_bearish and weekly_downtrend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: daily bearish crossover OR weekly trend turns down
            if daily_bearish or not weekly_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: daily bullish crossover OR weekly trend turns up
            if daily_bullish or not weekly_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals