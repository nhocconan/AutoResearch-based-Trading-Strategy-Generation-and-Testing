#!/usr/bin/env python3
"""
1d_WeeklyDonchian_Breakout_RSI_Filter_v1
Hypothesis: 1d Donchian breakout with weekly trend filter (price above/below 10-week SMA) and RSI(14) filter (RSI<70 for longs, RSI>30 for shorts) to avoid overextended entries. Works in bull (breakouts) and bear (fades from weekly extremes) by requiring alignment with weekly trend. Target: 20-60 trades over 4 years (5-15/year).
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
    
    # === Donchian Channel (20) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === RSI(14) ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # === Weekly 10 SMA for trend filter ===
    df_1w = get_htf_data(prices, '1w')
    sma_10_1w = pd.Series(df_1w['close'].values).rolling(window=10, min_periods=10).mean().values
    sma_10_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_10_1w)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(sma_10_1w_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price breaks above Donchian high, above weekly SMA10, RSI not overbought
            if (close[i] > highest_high[i] and 
                close[i] > sma_10_1w_aligned[i] and 
                rsi[i] < 70):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below Donchian low, below weekly SMA10, RSI not oversold
            elif (close[i] < lowest_low[i] and 
                  close[i] < sma_10_1w_aligned[i] and 
                  rsi[i] > 30):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price breaks below Donchian low OR RSI overbought
            if (close[i] < lowest_low[i] or 
                rsi[i] > 70):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above Donchian high OR RSI oversold
            if (close[i] > highest_high[i] or 
                rsi[i] < 30):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyDonchian_Breakout_RSI_Filter_v1"
timeframe = "1d"
leverage = 1.0