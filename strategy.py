#!/usr/bin/env python3
"""
1d_Williams_Alligator_1wTrend_Filter_v1
Hypothesis: Williams Alligator (jaw/teeth/lips) on 1d with 1w EMA50 trend filter to capture sustained trends while avoiding whipsaws in choppy markets. Uses discrete sizing (0.25) for ~15 trades/year. Alligator signals: lips > teeth > jaw = uptrend; lips < teeth < jaw = downtrend. 1w EMA50 ensures alignment with weekly trend, improving bear market performance by filtering counter-trend signals. Designed for BTC/ETH robustness with minimal overtrading.
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
    
    # Get 1d data for Alligator calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    median_price_1d = (high_1d + low_1d + close_1d) / 3.0
    
    # Williams Alligator: jaw (13,8), teeth (8,5), lips (5,3) SMAs of median price
    jaw = pd.Series(median_price_1d).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(median_price_1d).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(median_price_1d).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Alligator (max shift 8) and EMA50
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema50_1w_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Alligator signals: lips > teeth > jaw = uptrend; lips < teeth < jaw = downtrend
            # Filter by 1w EMA50 trend: only long above EMA50, short below EMA50
            long_signal = (lips[i] > teeth[i] > jaw[i]) and (close[i] > ema50_1w_aligned[i])
            short_signal = (lips[i] < teeth[i] < jaw[i]) and (close[i] < ema50_1w_aligned[i])
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit when Alligator reverses: lips < teeth or teeth < jaw
            exit_signal = (lips[i] < teeth[i]) or (teeth[i] < jaw[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when Alligator reverses: lips > teeth or teeth > jaw
            exit_signal = (lips[i] > teeth[i]) or (teeth[i] > jaw[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Williams_Alligator_1wTrend_Filter_v1"
timeframe = "1d"
leverage = 1.0