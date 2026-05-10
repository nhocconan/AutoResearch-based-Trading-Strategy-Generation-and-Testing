#!/usr/bin/env python3
# 12H_Trix_Signal_Crossover_With_Volume_and_Trend_Filter
# Hypothesis: Uses TRIX (triple exponential average) on 12h for momentum, with volume confirmation and 1d trend filter.
# Enters long when TRIX crosses above zero with volume > 1.5x average and price above 1d EMA50.
# Enters short when TRIX crosses below zero with volume > 1.5x average and price below 1d EMA50.
# Exits when TRIX crosses back through zero or volume drops below threshold.
# Uses 1d EMA50 for trend to avoid whipsaws in choppy markets.
# Targets 15-35 trades per year on 12h timeframe with position size 0.25 to minimize fee drag.

name = "12H_Trix_Signal_Crossover_With_Volume_and_Trend_Filter"
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
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend direction
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate TRIX on 12h: triple EMA of price, then rate of change
    # TRIX = EMA(EMA(EMA(close), period), period), period) then ROC
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = ema2.ewm(span=15, adjust=False, min_periods=15).mean()
    trix = ema3.pct_change(periods=1) * 100  # Percentage change
    trix_values = trix.values
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(45, 20)  # Warmup for TRIX and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(trix_values[i]) or np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_threshold[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below 1d EMA50
        price_above_ema = close[i] > ema_50_1d_aligned[i]
        price_below_ema = close[i] < ema_50_1d_aligned[i]
        
        # Volume condition
        vol_ok = volume[i] > vol_threshold[i]
        
        if position == 0:
            # Long entry: TRIX crosses above zero with volume and uptrend
            if (trix_values[i] > 0 and trix_values[i-1] <= 0 and 
                vol_ok and price_above_ema):
                signals[i] = 0.25
                position = 1
            # Short entry: TRIX crosses below zero with volume and downtrend
            elif (trix_values[i] < 0 and trix_values[i-1] >= 0 and 
                  vol_ok and price_below_ema):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: TRIX crosses below zero or volume drops
            if (trix_values[i] < 0 and trix_values[i-1] >= 0) or not vol_ok:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: TRIX crosses above zero or volume drops
            if (trix_values[i] > 0 and trix_values[i-1] <= 0) or not vol_ok:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals