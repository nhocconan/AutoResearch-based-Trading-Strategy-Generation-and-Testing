#!/usr/bin/env python3
"""
1d_ma_crossover_v1
Hypothesis: Weekly trend + daily moving average crossover with volume confirmation.
- Only trade in direction of weekly trend (above/below weekly EMA)
- Long: Weekly bullish + daily SMA(10) crosses above SMA(20) + volume above average
- Short: Weekly bearish + daily SMA(10) crosses below SMA(20) + volume above average
- Exit on opposite crossover or weekly trend reversal
- Target: 20-30 trades/year to avoid overtrading
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_ma_crossover_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data (same as input)
    df_1d = prices.copy()
    
    # Daily SMAs
    close_series = pd.Series(close)
    sma_fast = close_series.rolling(window=10, min_periods=10).mean().values
    sma_slow = close_series.rolling(window=20, min_periods=20).mean().values
    
    # Daily average volume for confirmation
    vol_series = pd.Series(volume)
    vol_avg = vol_series.rolling(window=20, min_periods=20).mean().values
    
    # Weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    weekly_ema = pd.Series(close_1w).ewm(span=20, min_periods=20, adjust=False).mean().values
    weekly_bullish = close_1w > weekly_ema
    weekly_bearish = close_1w < weekly_ema
    
    # Align weekly trend to daily
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after SMA warmup
        # Skip if data not ready
        if (np.isnan(sma_fast[i]) or np.isnan(sma_slow[i]) or 
            np.isnan(vol_avg[i]) or np.isnan(weekly_bullish_aligned[i]) or 
            np.isnan(weekly_bearish_aligned[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long
            # Exit: opposite crossover or weekly turns bearish
            if sma_fast[i] <= sma_slow[i] or weekly_bearish_aligned[i] > 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: opposite crossover or weekly turns bullish
            if sma_fast[i] >= sma_slow[i] or weekly_bullish_aligned[i] > 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long: weekly bullish + bullish crossover + volume confirmation
            if (weekly_bullish_aligned[i] > 0.5 and 
                sma_fast[i-1] <= sma_slow[i-1] and sma_fast[i] > sma_slow[i] and
                volume[i] > vol_avg[i]):
                position = 1
                signals[i] = 0.25
            # Short: weekly bearish + bearish crossover + volume confirmation
            elif (weekly_bearish_aligned[i] > 0.5 and 
                  sma_fast[i-1] >= sma_slow[i-1] and sma_fast[i] < sma_slow[i] and
                  volume[i] > vol_avg[i]):
                position = -1
                signals[i] = -0.25
    
    return signals