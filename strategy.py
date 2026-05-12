#!/usr/bin/env python3
# 4h_TRIX_VolumeSpike_TrendFilter
# Hypothesis: On 4h timeframe, enter long when TRIX crosses above 0 with price > 100-period EMA and volume > 1.5x 20-period MA.
# Enter short when TRIX crosses below 0 with price < 100-period EMA and volume > 1.5x 20-period MA.
# Exit when TRIX crosses back below 0 (for longs) or above 0 (for shorts).
# TRIX captures momentum shifts, volume confirms institutional participation, EMA filter avoids countertrend trades.
# Designed to work in both bull and bear markets by following momentum with volume confirmation.
# Targets ~30-50 trades/year to minimize fee drag.

name = "4h_TRIX_VolumeSpike_TrendFilter"
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
    
    # Calculate TRIX: triple EMA of percent change
    # ROC = (close / close.shift(1) - 1) * 100
    roc = np.zeros_like(close)
    roc[1:] = (close[1:] / close[:-1] - 1) * 100
    roc[0] = 0
    
    # Triple EMA
    ema1 = pd.Series(roc).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    trix = ema3
    
    # 100-period EMA for trend filter
    ema100 = pd.Series(close).ewm(span=100, adjust=False, min_periods=100).mean().values
    
    # Volume confirmation: 20-period moving average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        trix_now = trix[i]
        trix_prev = trix[i-1]
        ema100_val = ema100[i]
        vol_ma_val = vol_ma[i]
        vol_now = volume[i]
        
        if position == 0:
            # LONG: TRIX crosses above 0 with price > EMA100 and volume spike
            if trix_now > 0 and trix_prev <= 0 and close[i] > ema100_val and vol_now > vol_ma_val * 1.5:
                signals[i] = 0.25
                position = 1
            # SHORT: TRIX crosses below 0 with price < EMA100 and volume spike
            elif trix_now < 0 and trix_prev >= 0 and close[i] < ema100_val and vol_now > vol_ma_val * 1.5:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TRIX crosses back below 0
            if trix_now < 0 and trix_prev >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TRIX crosses back above 0
            if trix_now > 0 and trix_prev <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals