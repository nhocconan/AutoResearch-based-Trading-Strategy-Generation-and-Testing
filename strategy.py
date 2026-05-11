#!/usr/bin/env python3
"""
4h_Momentum_Catch_v1
Hypothesis: Captures momentum bursts with volume confirmation during trend continuations.
Uses 4h price action: long when price breaks above recent high with rising volume and bullish trend,
short when price breaks below recent low with rising volume and bearish trend.
Designed to work in both bull and bear markets by following the trend.
Target: 20-50 trades per year on 4h timeframe.
"""

name = "4h_Momentum_Catch_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4H Data for Trend Filter ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    # 4h EMA50 for trend filter
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # === Price Channels (20-period) ===
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(lookback, n):
        highest_high[i] = np.max(high[i-lookback:i])
        lowest_low[i] = np.min(low[i-lookback:i])
    
    # === Volume Moving Average ===
    vol_ma = np.full(n, np.nan)
    vol_lookback = 20
    for i in range(vol_lookback, n):
        vol_ma[i] = np.mean(volume[i-vol_lookback:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(50, lookback, vol_lookback)
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above recent high with volume confirmation in uptrend
            if (close[i] > highest_high[i] and 
                volume[i] > vol_ma[i] * 1.5 and 
                ema50_4h_aligned[i] < close[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below recent low with volume confirmation in downtrend
            elif (close[i] < lowest_low[i] and 
                  volume[i] > vol_ma[i] * 1.5 and 
                  ema50_4h_aligned[i] > close[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price falls below recent low OR trend turns bearish
            if (close[i] < lowest_low[i] or 
                ema50_4h_aligned[i] > close[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price rises above recent high OR trend turns bullish
            if (close[i] > highest_high[i] or 
                ema50_4h_aligned[i] < close[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals