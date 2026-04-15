#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h EMA pullback strategy with 1d trend filter
# Uses 6h EMA(20) as dynamic support/resistance with 1d EMA(50) trend filter.
# Long: price pulls back to EMA20 in uptrend (price > EMA50_1d). Short: price rallies to EMA20 in downtrend (price < EMA50_1d).
# Works in trending markets by buying dips and selling rallies. Target: 50-150 total trades.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate EMA(50) on 1d
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate EMA(20) on 6h
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if np.isnan(ema50_1d_aligned[i]):
            continue
        
        # Long: price pulls back to EMA20 in uptrend
        if (close[i] > ema50_1d_aligned[i] and  # Uptrend filter
            low[i] <= ema20[i] and             # Pullback to EMA20
            close[i] > ema20[i] and            # Close above EMA20 (confirm bounce)
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short: price rallies to EMA20 in downtrend
        elif (close[i] < ema50_1d_aligned[i] and  # Downtrend filter
              high[i] >= ema20[i] and             # Rally to EMA20
              close[i] < ema20[i] and             # Close below EMA20 (confirm rejection)
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: trend reversal or opposite signal
        elif position == 1 and close[i] < ema50_1d_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > ema50_1d_aligned[i]:
            position = 0
            signals[i] = 0.0
    
    return signals

name = "6h_EMA_Pullback_1dTrend"
timeframe = "6h"
leverage = 1.0