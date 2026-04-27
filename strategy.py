#!/usr/bin/env python3
"""
4h_MACD_Hist_ZeroCross_Volume_Trend
Hypothesis: MACD histogram zero-cross on 4h with volume confirmation and 12h trend filter.
- Long: MACD histogram crosses above zero with volume spike (>1.5x 20-bar avg) and 12h uptrend (close > EMA50)
- Short: MACD histogram crosses below zero with volume spike and 12h downtrend (close < EMA50)
- Exit: Opposite MACD zero-cross or trend failure
- Designed to capture momentum shifts with institutional volume confirmation
- Target: 20-50 trades/year (80-200 total over 4 years)
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
    
    # MACD calculation (12,26,9)
    close_s = pd.Series(close)
    ema12 = close_s.ewm(span=12, adjust=False, min_periods=12).mean().values
    ema26 = close_s.ewm(span=26, adjust=False, min_periods=26).mean().values
    macd_line = ema12 - ema26
    signal_line = pd.Series(macd_line).ewm(span=9, adjust=False, min_periods=9).mean().values
    macd_hist = macd_line - signal_line
    
    # 12h trend filter (EMA50)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    ema50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume spike detection (20-period average)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (vol_ma * 1.5)  # Volume at least 1.5x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough data for calculations
    start_idx = max(26, 9, 50) + 1
    
    for i in range(start_idx, n):
        if (np.isnan(macd_hist[i]) or np.isnan(macd_hist[i-1]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: MACD hist crosses above zero with volume spike and 12h uptrend
            if (macd_hist[i-1] <= 0 and macd_hist[i] > 0 and volume_spike[i] and close[i] > ema50_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: MACD hist crosses below zero with volume spike and 12h downtrend
            elif (macd_hist[i-1] >= 0 and macd_hist[i] < 0 and volume_spike[i] and close[i] < ema50_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: MACD hist crosses below zero or trend failure
            if (macd_hist[i-1] >= 0 and macd_hist[i] < 0) or close[i] < ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: MACD hist crosses above zero or trend failure
            if (macd_hist[i-1] <= 0 and macd_hist[i] > 0) or close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_MACD_Hist_ZeroCross_Volume_Trend"
timeframe = "4h"
leverage = 1.0