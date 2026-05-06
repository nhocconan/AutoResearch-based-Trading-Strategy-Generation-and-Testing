#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1-week 200-period EMA for trend and 1-day Williams %R for mean-reversion entries
# Long when price is above weekly EMA200 (bullish trend) and daily Williams %R crosses above -80 (oversold bounce)
# Short when price is below weekly EMA200 (bearish trend) and daily Williams %R crosses below -20 (overbought rejection)
# Uses Williams %R for precise entries within trend, targeting 20-40 trades per year (80-160 over 4 years) with 0.25 position sizing

name = "12h_1wEMA200_1dWilliamsR_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate weekly EMA200 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema200_1w = pd.Series(close_1w).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Calculate daily Williams %R (14-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_1d) / (highest_high - lowest_low) * -100
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # avoid division by zero
    
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):  # Start after weekly EMA200 warmup
        # Skip if any critical value is NaN or outside session
        if (np.isnan(ema200_1w_aligned[i]) or np.isnan(williams_r_aligned[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: above weekly EMA200 and Williams %R crosses above -80 (oversold bounce)
            if (close[i] > ema200_1w_aligned[i] and 
                williams_r_aligned[i] > -80 and 
                williams_r_aligned[i-1] <= -80):
                signals[i] = 0.25
                position = 1
            # Short: below weekly EMA200 and Williams %R crosses below -20 (overbought rejection)
            elif (close[i] < ema200_1w_aligned[i] and 
                  williams_r_aligned[i] < -20 and 
                  williams_r_aligned[i-1] >= -20):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below weekly EMA200 (trend change)
            if close[i] < ema200_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above weekly EMA200 (trend change)
            if close[i] > ema200_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals