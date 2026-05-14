#!/usr/bin/env python3
"""
12H_1D_200EMA_Trend_With_Volume_Filter
Hypothesis: On 12h timeframe, use daily 200 EMA as trend filter and 12h price action for entry.
In bull markets: price above daily 200 EMA, look for long entries when 12h closes above 12h EMA50 with volume.
In bear markets: price below daily 200 EMA, look for short entries when 12h closes below 12h EMA50 with volume.
This avoids counter-trend trades and uses volume to confirm momentum. Targets 12-30 trades/year.
"""
name = "12H_1D_200EMA_Trend_With_Volume_Filter"
timeframe = "12h"
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
    
    # Get 1D data for 200 EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Calculate 200 EMA on daily close
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate 12h EMA50 for entry signal
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume filter: current 12h volume > 1.5 x 20-period average volume
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(200, 50)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(ema_50[i]) or 
            np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above daily 200 EMA, 12h close above EMA50, and volume confirmation
            if (close[i] > ema_200_1d_aligned[i] and 
                close[i] > ema_50[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below daily 200 EMA, 12h close below EMA50, and volume confirmation
            elif (close[i] < ema_200_1d_aligned[i] and 
                  close[i] < ema_50[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below EMA50
            if close[i] < ema_50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above EMA50
            if close[i] > ema_50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals