#!/usr/bin/env python3
"""
6h_TRIX_Trend_Reversal_v1
Hypothesis: TRIX on 1d filters trend direction, TRIX on 6h triggers reversals at extremes. Long when 1d TRIX>0 and 6h TRIX crosses above -0.15; short when 1d TRIX<0 and 6h TRIX crosses below +0.15. Works in bull/bear by following higher timeframe momentum while catching mean-reversion swings within the trend. Target: 15-30 trades per year on 6h.
"""

name = "6h_TRIX_Trend_Reversal_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h TRIX for entry signals ===
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean()
    trix_6h = ((ema3 - ema3.shift(1)) / ema3.shift(1) * 100).fillna(0).values
    
    # === 1d TRIX for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema1_1d = pd.Series(close_1d).ewm(span=12, adjust=False, min_periods=12).mean()
    ema2_1d = pd.Series(ema1_1d).ewm(span=12, adjust=False, min_periods=12).mean()
    ema3_1d = pd.Series(ema2_1d).ewm(span=12, adjust=False, min_periods=12).mean()
    trix_1d = ((ema3_1d - ema3_1d.shift(1)) / ema3_1d.shift(1) * 100).fillna(0).values
    trix_1d_aligned = align_htf_to_ltf(prices, df_1d, trix_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 15
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if np.isnan(trix_1d_aligned[i]) or np.isnan(trix_6h[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: 1d bullish (TRIX>0) and 6h TRIX crosses above -0.15 from below
            if trix_1d_aligned[i] > 0 and trix_6h[i-1] <= -0.15 and trix_6h[i] > -0.15:
                signals[i] = 0.25
                position = 1
            # Short: 1d bearish (TRIX<0) and 6h TRIX crosses below +0.15 from above
            elif trix_1d_aligned[i] < 0 and trix_6h[i-1] >= 0.15 and trix_6h[i] < 0.15:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: 1d turns bearish OR 6h TRIX crosses below -0.15
            if trix_1d_aligned[i] < 0 or (trix_6h[i-1] > -0.15 and trix_6h[i] <= -0.15):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: 1d turns bullish OR 6h TRIX crosses above +0.15
            if trix_1d_aligned[i] > 0 or (trix_6h[i-1] < 0.15 and trix_6h[i] >= 0.15):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals