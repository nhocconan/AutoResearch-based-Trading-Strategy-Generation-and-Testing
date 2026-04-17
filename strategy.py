#!/usr/bin/env python3
"""
1h_Aroon_Slope_Trend_v1
Aroon(25) for trend strength: Aroon-Up > 60 for long, Aroon-Down > 60 for short.
1d EMA50 filter: price above/below 1d EMA50 for trend alignment.
Session filter: 08-20 UTC to avoid low-volume periods.
Target: 60-150 total trades over 4 years (15-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # === Aroon(25) ===
    # Aroon-Up: ((25 - periods since 25-period high) / 25) * 100
    # Aroon-Down: ((25 - periods since 25-period low) / 25) * 100
    period = 25
    aroon_up = np.full(n, np.nan)
    aroon_down = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        # Find highest high in last 'period' bars
        highest_high_idx = i - np.argmax(high[i - period + 1:i + 1][::-1])  # most recent
        # Find lowest low in last 'period' bars
        lowest_low_idx = i - np.argmin(low[i - period + 1:i + 1][::-1])    # most recent
        
        aroon_up[i] = ((period - (i - highest_high_idx)) / period) * 100
        aroon_down[i] = ((period - (i - lowest_low_idx)) / period) * 100
    
    # === 1d EMA50 for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 60
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(aroon_up[i]) or 
            np.isnan(aroon_down[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Session filter: only trade during 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: Aroon-Up > 60, Aroon-Down < 40 (strong uptrend), price above 1d EMA50
            if (aroon_up[i] > 60 and 
                aroon_down[i] < 40 and 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.20
                position = 1
                continue
            # Short: Aroon-Down > 60, Aroon-Up < 40 (strong downtrend), price below 1d EMA50
            elif (aroon_down[i] > 60 and 
                  aroon_up[i] < 40 and 
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.20
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: Aroon-Down > 60 (downtrend emerging) OR Aroon-Up < 40 (weakening uptrend)
            if (aroon_down[i] > 60 or 
                aroon_up[i] < 40):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: Aroon-Up > 60 (uptrend emerging) OR Aroon-Down < 40 (weakening downtrend)
            if (aroon_up[i] > 60 or 
                aroon_down[i] < 40):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Aroon_Slope_Trend_v1"
timeframe = "1h"
leverage = 1.0