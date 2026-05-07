#!/usr/bin/env python3
"""
6h_EMA20_1dTrend_12hVolume_Signal
Hypothesis: On 6b, enter long when price crosses above EMA20 with 1d uptrend (price>EMA50) and 12h volume above 1.5x average; short when price crosses below EMA20 with 1d downtrend and 12h volume spike. Uses EMA20 for responsiveness, 1d trend for bias, and volume for confirmation. Designed for 6h to achieve 15-30 trades/year with clear trend following logic that works in both bull (follow uptrend) and bear (follow downtrend) markets.
"""
name = "6h_EMA20_1dTrend_12hVolume_Signal"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get 12h data for volume filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate EMA20 on 6b
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate 1-day EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 12h volume average
    vol_avg_12h = pd.Series(df_12h['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_avg_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_avg_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Need sufficient warmup for averages
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(ema_20[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_avg_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price crosses above EMA20 + 1d uptrend + 12h volume spike
            if (close[i] > ema_20[i] and close[i-1] <= ema_20[i-1] and  # crossover
                close[i] > ema_50_1d_aligned[i] and  # 1d uptrend
                volume[i] > vol_avg_12h_aligned[i] * 1.5):  # volume spike
                signals[i] = 0.25
                position = 1
            # Short: price crosses below EMA20 + 1d downtrend + 12h volume spike
            elif (close[i] < ema_20[i] and close[i-1] >= ema_20[i-1] and  # crossover
                  close[i] < ema_50_1d_aligned[i] and  # 1d downtrend
                  volume[i] > vol_avg_12h_aligned[i] * 1.5):  # volume spike
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: price returns to EMA20 (mean reversion within trend)
            if position == 1:
                if close[i] < ema_20[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > ema_20[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals