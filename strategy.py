#!/usr/bin/env python3
name = "1d_StochasticOscillator_1wTrend_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 14:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w trend filter: EMA20
    df_1w = get_htf_data(prices, '1w')
    ema20_1w = pd.Series(df_1w['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Stochastic Oscillator (14,3,3)
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low)
    k_percent = np.where((highest_high - lowest_low) == 0, 50, k_percent)
    d_percent = pd.Series(k_percent).rolling(window=3, min_periods=3).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 16  # need 14 for Stoch + 3 for D
    
    for i in range(start_idx, n):
        # Skip if 1w trend data not ready
        if np.isnan(ema20_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Stochastic oversold (<20) and %K crosses above %D + 1w uptrend
            if (k_percent[i-1] < 20 and k_percent[i] >= 20 and  # exited oversold
                k_percent[i] > d_percent[i] and                  # bullish crossover
                close[i] > ema20_1w_aligned[i]):                 # 1w uptrend filter
                signals[i] = 0.25
                position = 1
            # Short: Stochastic overbought (>80) and %K crosses below %D + 1w downtrend
            elif (k_percent[i-1] > 80 and k_percent[i] <= 80 and  # exited overbought
                  k_percent[i] < d_percent[i] and                 # bearish crossover
                  close[i] < ema20_1w_aligned[i]):                # 1w downtrend filter
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when Stochastic becomes overbought (>80) or trend changes
            if (k_percent[i] > 80 or close[i] < ema20_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when Stochastic becomes oversold (<20) or trend changes
            if (k_percent[i] < 20 or close[i] > ema20_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals