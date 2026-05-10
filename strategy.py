#!/usr/bin/env python3
# 4h_TRIX_Volume_Spike_With_Trend_Filter
# Hypothesis: TRIX (TRIple Exponential Average) on 4h filters noise and captures momentum.
# Long when TRIX crosses above zero + 1d trend up + volume spike; short when TRIX crosses below zero + 1d trend down + volume spike.
# Uses volume confirmation to avoid false signals and 1d trend for multi-timeframe alignment.
# Designed for moderate trade frequency (target: 20-50 trades/year) with strong momentum signals.

name = "4h_TRIX_Volume_Spike_With_Trend_Filter"
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
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate TRIX on 4h close (15-period)
    # TRIX = EMA(EMA(EMA(close, 15), 15), 15)
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = ema2.ewm(span=15, adjust=False, min_periods=15).mean()
    # TRIX = (ema3 - ema3.shift(1)) / ema3.shift(1) * 100
    trix = (ema3 - ema3.shift(1)) / ema3.shift(1) * 100
    trix = trix.fillna(0).values
    
    # Calculate daily EMA for trend filter (34-period)
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation (20-period MA on 4h chart)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need TRIX (need 15*3=45 for triple EMA, 34 for EMA, 20 for volume)
    start_idx = max(45, 34, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(trix[i]) or np.isnan(trix[i-1]) or 
            np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # TRIX zero crossover
        trix_cross_up = trix[i-1] <= 0 and trix[i] > 0
        trix_cross_down = trix[i-1] >= 0 and trix[i] < 0
        
        # Daily trend filter
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        if position == 0:
            # Long entry: TRIX crosses up + daily uptrend + volume spike
            if trix_cross_up and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: TRIX crosses down + daily downtrend + volume spike
            elif trix_cross_down and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: TRIX crosses below zero or daily trend turns down
            if trix[i] < 0 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: TRIX crosses above zero or daily trend turns up
            if trix[i] > 0 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals