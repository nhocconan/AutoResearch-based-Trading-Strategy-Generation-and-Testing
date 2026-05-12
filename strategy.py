#!/usr/bin/env python3
name = "4h_RVOL_Trend_Reversal"
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
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d Trend Filter ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema100_1d = pd.Series(close_1d).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema100_1d_aligned = align_htf_to_ltf(prices, df_1d, ema100_1d)
    
    # === Relative Volume (RVOL) ===
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    rvol = volume / vol_avg_20  # RVOL > 1.5 = volume spike
    
    # === 4h Price Momentum (ROC 3-period) ===
    roc3 = np.zeros_like(close)
    roc3[3:] = (close[3:] - close[:-3]) / close[:-3] * 100
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(ema100_1d_aligned[i]) or np.isnan(rvol[i]) or np.isnan(roc3[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price above 1d EMA100, positive momentum, and volume spike
            if (close[i] > ema100_1d_aligned[i] and 
                roc3[i] > 0.1 and 
                rvol[i] > 1.8):
                signals[i] = 0.25
                position = 1
            # Short: Price below 1d EMA100, negative momentum, and volume spike
            elif (close[i] < ema100_1d_aligned[i] and 
                  roc3[i] < -0.1 and 
                  rvol[i] > 1.8):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price crosses below 1d EMA100 OR momentum turns negative
            if close[i] < ema100_1d_aligned[i] or roc3[i] < -0.05:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price crosses above 1d EMA100 OR momentum turns positive
            if close[i] > ema100_1d_aligned[i] or roc3[i] > 0.05:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals