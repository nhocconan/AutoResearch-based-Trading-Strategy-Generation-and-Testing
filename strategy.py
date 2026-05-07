#!/usr/bin/env python3
"""
4h_CombinedTrendAndMomentum_v1
Hypothesis: Combines trend (21-period Exponential Moving Average) and momentum (Rate of Change over 5 periods) on the 4h timeframe.
Requires both indicators to agree on direction, filtered by volume above average and confirmed by 1-day trend (EMA-20).
Designed to work in both bull and bear markets by using symmetric long/short logic.
Target: 20-40 trades per year to minimize fee drag.
"""

name = "4h_CombinedTrendAndMomentum_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 21-period EMA for trend
    close_series = pd.Series(close)
    ema21 = close_series.ewm(span=21, adjust=False, min_periods=21).values
    
    # 5-period ROC for momentum
    roc = np.zeros_like(close)
    for i in range(5, n):
        if close[i-5] != 0:
            roc[i] = (close[i] - close[i-5]) / close[i-5]
    
    # Volume confirmation: current volume > 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 1-day trend filter: EMA-20 on daily close
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema1d_aligned = align_htf_to_ltf(prices, df_1d, ema1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(21, n):
        # Skip if any critical value is NaN
        if (np.isnan(ema21[i]) or np.isnan(roc[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] == 0 or
            np.isnan(ema1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: EMA uptrend AND positive ROC with volume confirmation and 1-day uptrend
            if close[i] > ema21[i] and roc[i] > 0 and volume[i] > vol_ma[i] and close[i] > ema1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: EMA downtrend AND negative ROC with volume confirmation and 1-day downtrend
            elif close[i] < ema21[i] and roc[i] < 0 and volume[i] > vol_ma[i] and close[i] < ema1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: EMA downtrend OR negative ROC
            if close[i] < ema21[i] or roc[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: EMA uptrend OR positive ROC
            if close[i] > ema21[i] or roc[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals