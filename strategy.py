#!/usr/bin/env python3
# 12h_ema_crossover_volume_v1
# Hypothesis: Uses 12h EMA crossovers with volume confirmation on 12h timeframe.
# Long when fast EMA crosses above slow EMA with volume > 1.5x average; short when fast EMA crosses below slow EMA.
# Includes volatility filter using ATR to avoid whipsaw in choppy markets.
# Designed to work in both bull and bear markets by capturing trend changes with volume confirmation.
# Target: 15-25 trades/year (60-100 total over 4 years) with strict entry conditions.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_ema_crossover_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1. EMA crossovers - 9 and 21 period
    ema9 = np.zeros(n)
    ema21 = np.zeros(n)
    
    # Initialize EMAs
    ema9[0] = close[0]
    ema21[0] = close[0]
    
    alpha9 = 2.0 / (9 + 1)
    alpha21 = 2.0 / (21 + 1)
    
    for i in range(1, n):
        ema9[i] = alpha9 * close[i] + (1 - alpha9) * ema9[i-1]
        ema21[i] = alpha21 * close[i] + (1 - alpha21) * ema21[i-1]
    
    # 2. Volume confirmation - 20 period average
    vol_ma_20 = np.zeros(n)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma_20[i] = vol_sum / 20
    
    # 3. ATR for volatility filter
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = np.zeros(n)
    atr[0] = tr[0]
    for i in range(1, n):
        atr[i] = 0.05 * tr[i] + 0.95 * atr[i-1]  # Wilder's smoothing
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(21, n):  # Start after warmup
        # Skip if any required data is invalid
        if np.isnan(ema9[i]) or np.isnan(ema21[i]) or np.isnan(vol_ma_20[i]) or np.isnan(atr[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_ok = volume[i] > vol_ma_20[i] * 1.5
        
        # Volatility filter: avoid trading when ATR is too low (choppy market)
        vol_filter_ok = atr[i] > np.mean(atr[max(0, i-20):i+1]) * 0.8
        
        if position == 1:  # Long position
            # Exit: fast EMA crosses below slow EMA
            if ema9[i] < ema21[i] and ema9[i-1] >= ema21[i-1]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: fast EMA crosses above slow EMA
            if ema9[i] > ema21[i] and ema9[i-1] <= ema21[i-1]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: fast EMA crosses above slow EMA with volume and volatility filters
            if ema9[i] > ema21[i] and ema9[i-1] <= ema21[i-1] and vol_ok and vol_filter_ok:
                position = 1
                signals[i] = 0.25
            # Enter short: fast EMA crosses below slow EMA with volume and volatility filters
            elif ema9[i] < ema21[i] and ema9[i-1] >= ema21[i-1] and vol_ok and vol_filter_ok:
                position = -1
                signals[i] = -0.25
    
    return signals