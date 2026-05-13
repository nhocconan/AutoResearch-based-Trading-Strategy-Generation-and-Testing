#!/usr/bin/env python3
"""
4h_TRIX_ZeroLine_Cross_12hTrend_Volume
Hypothesis: Use TRIX(12) zero-line cross on 4h for momentum signals, confirmed by 12h EMA25 trend direction and volume > 1.5x average. TRIX filters noise and captures momentum shifts; 12h trend avoids counter-trend trades; volume ensures conviction. Works in bull/bear by only taking signals aligned with higher timeframe trend.
"""

name = "4h_TRIX_ZeroLine_Cross_12hTrend_Volume"
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
    
    # TRIX calculation: EMA of EMA of EMA of log(close), then ROC
    log_close = np.log(close)
    ema1 = pd.Series(log_close).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    trix = 100 * (pd.Series(ema3).pct_change().values)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # 12h EMA25 for trend direction
    ema_25_12h = pd.Series(df_12h['close'].values).ewm(span=25, adjust=False, min_periods=25).mean().values
    ema_25_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_25_12h)
    
    # Volume average (20-period) for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if any required data is NaN
        if (np.isnan(trix[i]) or np.isnan(ema_25_12h_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: TRIX crosses above zero, price above 12h EMA25, volume above average
            if trix[i] > 0 and trix[i-1] <= 0 and close[i] > ema_25_12h_aligned[i] and volume[i] > vol_ma[i] * 1.5:
                signals[i] = 0.25
                position = 1
            # SHORT: TRIX crosses below zero, price below 12h EMA25, volume above average
            elif trix[i] < 0 and trix[i-1] >= 0 and close[i] < ema_25_12h_aligned[i] and volume[i] > vol_ma[i] * 1.5:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TRIX crosses below zero or price breaks below 12h EMA25
            if trix[i] < 0 and trix[i-1] >= 0 or close[i] < ema_25_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TRIX crosses above zero or price breaks above 12h EMA25
            if trix[i] > 0 and trix[i-1] <= 0 or close[i] > ema_25_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals