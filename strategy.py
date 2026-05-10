#!/usr/bin/env python3
# 6h_MACD_Histogram_Zero_Cross_With_1d_Trend_Filter
# Hypothesis: MACD histogram crossing zero on 6h indicates momentum shift, filtered by 1d EMA trend direction.
# Long when MACD histogram crosses above zero and price > 1d EMA; short when crosses below zero and price < 1d EMA.
# Uses volume confirmation to filter false signals. Designed for low-to-moderate trade frequency (15-30 trades/year).
# Works in bull markets via momentum continuation and in bear markets via trend-following shorts.

name = "6h_MACD_Histogram_Zero_Cross_With_1d_Trend_Filter"
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
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate MACD (12,26,9) on 6h close
    ema12 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema26 = pd.Series(close).ewm(span=26, adjust=False, min_periods=26).mean().values
    macd_line = ema12 - ema26
    signal_line = pd.Series(macd_line).ewm(span=9, adjust=False, min_periods=9).mean().values
    macd_hist = macd_line - signal_line
    
    # Volume confirmation (20-period MA)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need MACD (26 for EMA26, 9 for signal) and volume MA
    start_idx = max(26, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(macd_hist[i]) or np.isnan(macd_hist[i-1]) or
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Daily trend filter
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        # MACD histogram zero cross
        hist_cross_up = macd_hist[i-1] <= 0 and macd_hist[i] > 0
        hist_cross_down = macd_hist[i-1] >= 0 and macd_hist[i] < 0
        
        if position == 0:
            # Long entry: MACD hist crosses up + uptrend + volume spike
            if hist_cross_up and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: MACD hist crosses down + downtrend + volume spike
            elif hist_cross_down and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: MACD hist crosses below zero or trend turns down
            if macd_hist[i] < 0 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: MACD hist crosses above zero or trend turns up
            if macd_hist[i] > 0 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals