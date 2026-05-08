# 4h_TRIX_Momentum_Trend_Filter
# Hypothesis: TRIX (12-period) captures momentum shifts with smoothing, while a 4h EMA50 trend filter ensures
# trades align with the dominant direction. Volume confirmation reduces false signals. Designed for 4h to
# limit trades (~20-40/year) and avoid fee drag. Works in bull/bear by following trend.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_TRIX_Momentum_Trend_Filter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h EMA50 for trend filter
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # TRIX: triple EMA of ROC, period=12
    # ROC = (close / close.shift(1) - 1) * 100
    roc = np.zeros(n)
    roc[1:] = (close[1:] / close[:-1] - 1) * 100
    
    # First EMA of ROC
    ema1 = pd.Series(roc).ewm(span=12, adjust=False, min_periods=12).mean().values
    # Second EMA
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    # Third EMA = TRIX
    trix = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # max of 12*3 (TRIX) and 20 (vol)
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if np.isnan(trix[i]) or np.isnan(ema_50[i]) or np.isnan(volume_confirm[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: TRIX crosses above zero with uptrend + volume
            long_cond = (trix[i] > 0 and trix[i-1] <= 0 and 
                        ema_50[i] > ema_50[i-1] and
                        volume_confirm[i])
            
            # Short: TRIX crosses below zero with downtrend + volume
            short_cond = (trix[i] < 0 and trix[i-1] >= 0 and 
                         ema_50[i] < ema_50[i-1] and
                         volume_confirm[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: TRIX crosses below zero OR trend breaks
            if trix[i] < 0 or ema_50[i] < ema_50[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: TRIX crosses above zero OR trend breaks
            if trix[i] > 0 or ema_50[i] > ema_50[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals