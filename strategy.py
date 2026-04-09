#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Elder Ray (Bull/Bear Power) + 1w trend filter
# Elder Ray measures bull/bear power relative to EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13
# Long when: Bull Power > 0 AND Bear Power rising (less negative) AND 1w close > 1w EMA34 (uptrend)
# Short when: Bear Power < 0 AND Bull Power falling (less positive) AND 1w close < 1w EMA34 (downtrend)
# Uses discrete position sizing 0.25 to target ~12-30 trades/year and minimize fee drag
# Works in bull/bear markets: trend filter ensures we only trade with higher timeframe momentum

name = "6h_1d_1w_elder_ray_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA13 for Elder Ray
    close_s_1d = pd.Series(close_1d)
    ema13_1d = close_s_1d.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate 1d Elder Ray components
    bull_power_1d = high_1d - ema13_1d  # Bull Power = High - EMA13
    bear_power_1d = low_1d - ema13_1d   # Bear Power = Low - EMA13
    
    # Calculate 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 40:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    close_s_1w = pd.Series(close_1w)
    ema34_1w = close_s_1w.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d and 1w indicators to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(ema34_1w_aligned[i]) or np.isnan(close_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit conditions: bear power turns negative OR trend turns down
            if bear_power_aligned[i] < 0 or close_1w_aligned[i] < ema34_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: bull power turns positive OR trend turns up
            if bull_power_aligned[i] > 0 or close_1w_aligned[i] > ema34_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry conditions: bull power positive AND rising (or bear power negative AND rising) with trend alignment
            bull_rising = bull_power_aligned[i] > bull_power_aligned[i-1] if i > 0 else False
            bear_rising = bear_power_aligned[i] > bear_power_aligned[i-1] if i > 0 else False  # less negative = rising
            
            if (bull_power_aligned[i] > 0 and bull_rising and close_1w_aligned[i] > ema34_1w_aligned[i]):
                position = 1
                signals[i] = 0.25
            elif (bear_power_aligned[i] < 0 and bear_rising and close_1w_aligned[i] < ema34_1w_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals