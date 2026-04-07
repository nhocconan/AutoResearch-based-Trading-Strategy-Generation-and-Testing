#!/usr/bin/env python3
"""
1d_weekly_ema_trend_v2
Hypothesis: On daily timeframe, follow the weekly trend using EMA crossover with volume confirmation.
Enter long when daily EMA20 crosses above weekly EMA20 and volume is above average.
Enter short when daily EMA20 crosses below weekly EMA20 and volume is above average.
Weekly trend filter ensures we only trade in the direction of higher timeframe trend.
Designed for 1d timeframe to target 10-25 trades/year, minimizing fee drag.
Works in both bull and bear markets by following the weekly trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_weekly_ema_trend_v2"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily EMA20
    ema_20d = pd.Series(close).ewm(span=20, min_periods=20).mean().values
    
    # Weekly EMA20
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema_20w = pd.Series(close_1w).ewm(span=20, min_periods=20).mean().values
    ema_20w_aligned = align_htf_to_ltf(prices, df_1w, ema_20w)
    
    # Volume confirmation: volume > 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if data not available
        if (np.isnan(ema_20d[i]) or np.isnan(ema_20w_aligned[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            signals[i] = 0.0
            continue
        
        # Weekly trend direction (based on slope)
        weekly_uptrend = ema_20w_aligned[i] > ema_20w_aligned[i-1] if i > 0 else False
        weekly_downtrend = ema_20w_aligned[i] < ema_20w_aligned[i-1] if i > 0 else False
        
        # Daily EMA20 crossover signals
        ema_cross_up = ema_20d[i] > ema_20w_aligned[i] and ema_20d[i-1] <= ema_20w_aligned[i-1]
        ema_cross_down = ema_20d[i] < ema_20w_aligned[i] and ema_20d[i-1] >= ema_20w_aligned[i-1]
        
        if position == 1:  # Long position
            # Exit: EMA20 crosses below weekly EMA20 or weekly trend turns down
            if ema_cross_down or not weekly_uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: EMA20 crosses above weekly EMA20 or weekly trend turns up
            if ema_cross_up or not weekly_downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: EMA20 crosses above weekly EMA20 with volume confirmation and weekly uptrend
            if ema_cross_up and vol_ma[i] > 0 and volume[i] > vol_ma[i] and weekly_uptrend:
                position = 1
                signals[i] = 0.25
            # Short: EMA20 crosses below weekly EMA20 with volume confirmation and weekly downtrend
            elif ema_cross_down and vol_ma[i] > 0 and volume[i] > vol_ma[i] and weekly_downtrend:
                position = -1
                signals[i] = -0.25
    
    return signals