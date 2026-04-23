#!/usr/bin/env python3
"""
Hypothesis: 4h TRIX momentum with 12h EMA50 trend filter and volume spike confirmation.
- TRIX(12) captures smoothed rate of change, effective in ranging and trending markets
- 12h EMA50 as trend filter (long only above, short only below) to avoid whipsaw
- Volume > 2.0x 20-period average for confirmation (adjusts for 4h frequency)
- Position size: 0.25 discrete level to minimize fee churn
- Target: 30-60 trades/year on 4h timeframe (120-240 total over 4 years)
- Works in both bull/bear via trend filter + momentum confirmation
"""

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
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # TRIX calculation: triple EMA of ROC
    # ROC = (close - close.shift(12)) / close.shift(12)
    close_s = pd.Series(close)
    roc = close_s.pct_change(12)  # 12-period rate of change
    ema1 = roc.ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = ema1.ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = ema2.ewm(span=12, adjust=False, min_periods=12).mean()
    trix = ema3.values * 100  # scale for readability
    
    # 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(24, 50)  # ROC(12) needs 12+ for ROC, plus EMA warmup
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(trix[i]) or
            np.isnan(ema_50_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        # TRIX momentum signals
        trix_up = trix[i] > 0 and trix[i-1] <= 0  # TRIX crosses above zero
        trix_down = trix[i] < 0 and trix[i-1] >= 0  # TRIX crosses below zero
        
        if position == 0:
            # Long: TRIX crosses above zero AND price above 12h EMA50 AND volume confirmation
            if trix_up and close[i] > ema_50_12h_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below zero AND price below 12h EMA50 AND volume confirmation
            elif trix_down and close[i] < ema_50_12h_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: TRIX crosses below zero OR price crosses below 12h EMA50
            if trix_down or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: TRIX crosses above zero OR price crosses above 12h EMA50
            if trix_up or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_TRIX_Momentum_12hEMA50_VolumeSpike_Filter_v1"
timeframe = "4h"
leverage = 1.0