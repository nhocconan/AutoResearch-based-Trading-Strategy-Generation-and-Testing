#!/usr/bin/env python3
# 4h_TRIX_VolumeSpike_RegimeFilter
# Hypothesis: TRIX crossing zero indicates momentum shift. Combine with volume spike (>1.5x 20MA) and Choppiness Index regime filter (CHOP < 38.2 = trending) to enter in direction of momentum. Exit when TRIX crosses zero in opposite direction. Targets 25-35 trades/year for low fee drift.

name = "4h_TRIX_VolumeSpike_RegimeFilter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # TRIX: triple EMA of ROC
    roc = np.diff(close, prepend=close[0]) / close  # ROC(1)
    ema1 = pd.Series(roc).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    trix = ema3 * 100  # scale for readability
    
    # Choppiness Index (14-period)
    atr_list = []
    for i in range(n):
        if i == 0:
            tr = high[i] - low[i]
        else:
            tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        atr_list.append(tr)
    atr = np.array(atr_list)
    atr_sum = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(14)
    
    # Volume confirmation: 1.5x 20-period moving average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_ma)
    
    # Align TRIX, CHOP, volume spike to ensure no look-ahead (already LTF)
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if np.isnan(trix[i]) or np.isnan(chop[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: TRIX crosses above zero, volume spike, trending regime (CHOP < 38.2)
            if trix[i] > 0 and trix[i-1] <= 0 and vol_spike[i] and chop[i] < 38.2:
                signals[i] = 0.25
                position = 1
            # SHORT: TRIX crosses below zero, volume spike, trending regime (CHOP < 38.2)
            elif trix[i] < 0 and trix[i-1] >= 0 and vol_spike[i] and chop[i] < 38.2:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TRIX crosses below zero
            if trix[i] < 0 and trix[i-1] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TRIX crosses above zero
            if trix[i] > 0 and trix[i-1] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals