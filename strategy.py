#!/usr/bin/env python3
"""
Hypothesis: 6-hour Williams %R with 1-week trend filter.
Long when Williams %R < -80 (oversold) and 1-week EMA50 is rising.
Short when Williams %R > -20 (overbought) and 1-week EMA50 is falling.
Exit when Williams %R crosses back above -50 (for longs) or below -50 (for shorts).
Williams %R identifies extreme short-term reversals; 1-week EMA50 filters higher timeframe trend.
Designed for low trade frequency by requiring extreme readings + trend alignment.
Works in both bull and bear markets by trading mean reversion within the weekly trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load 1-week data for EMA50 trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Williams %R (14 periods)
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(ema50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R < -80 (oversold) and 1-week EMA50 rising
            if (williams_r[i] < -80 and 
                ema50_1w_aligned[i] > ema50_1w_aligned[i-1]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought) and 1-week EMA50 falling
            elif (williams_r[i] > -20 and 
                  ema50_1w_aligned[i] < ema50_1w_aligned[i-1]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Williams %R crosses back above -50
                if williams_r[i] > -50:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Williams %R crosses back below -50
                if williams_r[i] < -50:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_WilliamsR_1wEMA50_Trend"
timeframe = "6h"
leverage = 1.0