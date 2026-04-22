# 4H_4H_SMA_Crossover_Volume_Filter
# Strategy: 4h SMA crossover with volume confirmation and trend filter
# Hypothesis: SMA crossovers on 4h timeframe capture medium-term trends. Volume confirmation reduces false breakouts. Works in both bull and bear by following trend direction.
# Target: 75-200 total trades over 4 years (19-50/year) to minimize fee drag.

#!/usr/bin/env python3
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
    
    # Calculate SMA(50) and SMA(100) on close prices
    sma_50 = pd.Series(close).rolling(window=50, min_periods=50).mean().values
    sma_100 = pd.Series(close).rolling(window=100, min_periods=100).mean().values
    
    # Calculate volume average for confirmation
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if np.isnan(sma_50[i]) or np.isnan(sma_100[i]) or np.isnan(vol_avg_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: SMA(50) crosses above SMA(100) with volume confirmation
            if (sma_50[i] > sma_100[i] and sma_50[i-1] <= sma_100[i-1] and
                volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = 0.30
                position = 1
            # Short: SMA(50) crosses below SMA(100) with volume confirmation
            elif (sma_50[i] < sma_100[i] and sma_50[i-1] >= sma_100[i-1] and
                  volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = -0.30
                position = -1
        else:
            # Exit: Opposite crossover occurs
            if position == 1:
                if sma_50[i] < sma_100[i] and sma_50[i-1] >= sma_100[i-1]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.30
            else:  # position == -1
                if sma_50[i] > sma_100[i] and sma_50[i-1] <= sma_100[i-1]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.30
    
    return signals

name = "4H_4H_SMA_Crossover_Volume_Filter"
timeframe = "4h"
leverage = 1.0