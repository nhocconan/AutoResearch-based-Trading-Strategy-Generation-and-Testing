#!/usr/bin/env python3
"""
4h_1d_TRIX_Trend_With_Volume_Filter
Hypothesis: Use TRIX (triple exponential average) on 1d to determine trend direction on 4h.
Enter long when TRIX turns positive and volume > 1.5x 20-period average.
Enter short when TRIX turns negative and volume > 1.5x 20-period average.
Exit when TRIX crosses zero in opposite direction.
Uses volume confirmation to reduce false signals and improve win rate.
Target: 20-40 trades/year per symbol. Works in bull/bear by following higher timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data once for TRIX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 15:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate TRIX: EMA of EMA of EMA of log returns, then percent change
    # Using period 15 as standard
    ema1 = pd.Series(close_1d).ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean()
    # TRIX = (ema3 - previous ema3) / previous ema3 * 100
    trix_raw = (ema3 - ema3.shift(1)) / ema3.shift(1) * 100
    trix = trix_raw.values
    
    # Align TRIX to 4h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if TRIX not ready
        if np.isnan(trix_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 1.5 * vol_ma
        else:
            volume_ok = False
        
        # TRIX signal: positive = uptrend, negative = downtrend
        trix_pos = trix_aligned[i] > 0
        trix_neg = trix_aligned[i] < 0
        
        if position == 0:
            # Long conditions: TRIX positive + volume
            if trix_pos and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short conditions: TRIX negative + volume
            elif trix_neg and volume_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: TRIX turns negative
            if trix_neg:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: TRIX turns positive
            if trix_pos:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1d_TRIX_Trend_With_Volume_Filter"
timeframe = "4h"
leverage = 1.0