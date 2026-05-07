#!/usr/bin/env python3
# 6h_MovingAverage_Crossover_1dTrend_Filter
# Hypothesis: Uses 6-hour MA crossover (MA20 x MA50) filtered by 1-day trend (close > EMA34) with volume confirmation.
# Designed for 6h timeframe with moderate trade frequency (20-40/year) and strong performance in both bull and bear regimes.
# The 1d trend filter avoids counter-trend trades during strong trends, while MA crossover captures momentum.
# Volume confirmation reduces false signals. Target: 80-160 total trades over 4 years (20-40/year).

name = "6h_MovingAverage_Crossover_1dTrend_Filter"
timeframe = "6h"
leverage = 1.0

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
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1-day EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Moving averages on 6h timeframe
    ma20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    ma50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume filter: current volume > 1.5x average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Ensure we have MA50, MA20 and volume MA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(ma20[i]) or np.isnan(ma50[i]) or 
            np.isnan(ema34_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: spike confirmation
        volume_filter = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: MA20 crosses above MA50 + Uptrend (close > EMA34) + volume confirmation
            if (ma20[i] > ma50[i] and ma20[i-1] <= ma50[i-1] and
                close[i] > ema34_aligned[i] and
                volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: MA20 crosses below MA50 + Downtrend (close < EMA34) + volume confirmation
            elif (ma20[i] < ma50[i] and ma20[i-1] >= ma50[i-1] and
                  close[i] < ema34_aligned[i] and
                  volume_filter):
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: MA20 crosses back in opposite direction
            if (position == 1 and ma20[i] < ma50[i]) or \
               (position == -1 and ma20[i] > ma50[i]):
                signals[i] = 0.0
                position = 0
            else:
                # Maintain position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals