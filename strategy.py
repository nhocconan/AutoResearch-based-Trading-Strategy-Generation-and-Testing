#!/usr/bin/env python3
"""
12h_12h_EMA_Crossover_Trend
Hypothesis: Uses 12h EMA crossover (34/89) for trend direction, with 1d volume confirmation
and 1w trend filter to reduce false signals. Targets 15-30 trades/year to minimize
fee drift and perform well in both bull and bear markets by avoiding whipsaws.
"""

name = "12h_12h_EMA_Crossover_Trend"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h EMA crossover: fast EMA(34) and slow EMA(89)
    fast_ema = pd.Series(close).ewm(span=34, adjust=False, min_periods=34).mean().values
    slow_ema = pd.Series(close).ewm(span=89, adjust=False, min_periods=89).mean().values
    
    # 1d volume confirmation: current volume > 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 1w trend filter: EMA of weekly close
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(89, n):
        # Skip if any critical value is NaN
        if (np.isnan(fast_ema[i]) or np.isnan(slow_ema[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] == 0 or np.isnan(ema_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: fast EMA crosses above slow EMA with volume confirmation and 1w trend up
            if fast_ema[i] > slow_ema[i] and fast_ema[i-1] <= slow_ema[i-1] and volume[i] > vol_ma[i] and close[i] > ema_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: fast EMA crosses below slow EMA with volume confirmation and 1w trend down
            elif fast_ema[i] < slow_ema[i] and fast_ema[i-1] >= slow_ema[i-1] and volume[i] > vol_ma[i] and close[i] < ema_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: fast EMA crosses below slow EMA
            if fast_ema[i] < slow_ema[i] and fast_ema[i-1] >= slow_ema[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: fast EMA crosses above slow EMA
            if fast_ema[i] > slow_ema[i] and fast_ema[i-1] <= slow_ema[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals