#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour TRIX momentum with volume spike and daily trend filter.
# Long when TRIX crosses above zero with daily EMA(34) uptrend and volume spike.
# Short when TRIX crosses below zero with daily EMA(34) downtrend and volume spike.
# Uses TRIX (12-period EMA of EMA of EMA) for smooth momentum, reducing whipsaw.
# Daily trend filter ensures alignment with higher timeframe momentum.
# Volume spike confirms institutional participation.
# Designed for 4h timeframe to target 20-50 trades/year, avoiding excessive frequency.

name = "4h_TRIX_DailyTrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data once for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate daily EMA(34) for trend filter
    daily_close = df_1d['close'].values
    ema34_1d = pd.Series(daily_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate TRIX on 4h: triple EMA of close, then ROC
    # TRIX = EMA(EMA(EMA(close, 12), 12), 12) then 1-period ROC
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    trix = 100 * (ema3 - np.roll(ema3, 1)) / np.roll(ema3, 1)
    trix[0] = 0  # first value undefined
    
    # Volume spike: current volume > 2.0 * 20-period average on 4h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(trix[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema34_1d_val = ema34_1d_aligned[i]
        trix_val = trix[i]
        trix_prev = trix[i-1]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: TRIX crosses above zero + daily uptrend + volume spike
            if (trix_prev <= 0 and trix_val > 0 and 
                close[i] > ema34_1d_val and vol_spike):
                signals[i] = 0.25
                position = 1
            # Enter short: TRIX crosses below zero + daily downtrend + volume spike
            elif (trix_prev >= 0 and trix_val < 0 and 
                  close[i] < ema34_1d_val and vol_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: TRIX crosses below zero OR daily trend turns down
            if (trix_prev >= 0 and trix_val < 0) or close[i] < ema34_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: TRIX crosses above zero OR daily trend turns up
            if (trix_prev <= 0 and trix_val > 0) or close[i] > ema34_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals