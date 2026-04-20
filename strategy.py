#!/usr/bin/env python3
# 6h_200EMA_Trend_Pullback_Volume
# Hypothesis: In strong trends (price above/below 200 EMA), pullbacks to the 20 EMA with volume confirmation offer high-probability entries.
# The 200 EMA defines the trend, 20 EMA provides dynamic support/resistance, and volume confirms institutional interest.
# Works in bull markets (buying dips in uptrend) and bear markets (selling rallies in downtrend).
# Target: 20-40 trades/year to stay under 160 total over 4 years.

name = "6h_200EMA_Trend_Pullback_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 210:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 20 EMA and 200 EMA
    close_series = pd.Series(close)
    ema_20 = close_series.ewm(span=20, adjust=False, min_periods=20).values
    ema_200 = close_series.ewm(span=200, adjust=False, min_periods=200).values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Wait for 200 EMA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(ema_20[i]) or np.isnan(ema_200[i]) or np.isnan(volume_filter[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: uptrend (price > 200 EMA) + pullback to 20 EMA + volume
            if close[i] > ema_200[i] and close[i] <= ema_20[i] * 1.005 and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: downtrend (price < 200 EMA) + rally to 20 EMA + volume
            elif close[i] < ema_200[i] and close[i] >= ema_20[i] * 0.995 and volume_filter[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if trend breaks or reversal signals
            if close[i] < ema_200[i] or close[i] > ema_20[i] * 1.02:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if trend breaks or reversal signals
            if close[i] > ema_200[i] or close[i] < ema_20[i] * 0.98:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals