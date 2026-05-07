#!/usr/bin/env python3
name = "6h_MACD_Stochastic_Confluence"
timeframe = "6h"
leverage = 1.0

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
    volume = prices['volume'].values
    
    # Get 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 6h MACD (12,26,9)
    ema_fast = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean()
    ema_slow = pd.Series(close).ewm(span=26, adjust=False, min_periods=26).mean()
    macd_line = (ema_fast - ema_slow).values
    signal_line = pd.Series(macd_line).ewm(span=9, adjust=False, min_periods=9).mean().values
    macd_hist = macd_line - signal_line
    
    # Calculate 6h Stochastic (14,3,3)
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    k_percent = k_percent.fillna(0).values
    d_percent = pd.Series(k_percent).rolling(window=3, min_periods=3).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(macd_hist[i]) or 
            np.isnan(d_percent[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: MACD bullish crossover, Stochastic oversold, uptrend (price > EMA50)
            if (macd_hist[i] > 0 and macd_hist[i-1] <= 0 and  # MACD crossover up
                k_percent[i] < 30 and 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: MACD bearish crossover, Stochastic overbought, downtrend (price < EMA50)
            elif (macd_hist[i] < 0 and macd_hist[i-1] >= 0 and  # MACD crossover down
                  k_percent[i] > 70 and 
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: MACD bearish crossover
            if macd_hist[i] < 0 and macd_hist[i-1] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: MACD bullish crossover
            if macd_hist[i] > 0 and macd_hist[i-1] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals