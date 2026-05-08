#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_TRIX_10_Trend_Filter_12hV"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for TRIX and volume filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # TRIX: 3x EMA smoothing of 1-period ROC (standard 10 period)
    # TRIX = EMA(EMA(EMA(ROC, 10), 10), 10)
    roc = np.diff(np.log(close_12h), prepend=np.log(close_12h[0])) * 100
    ema1 = pd.Series(roc).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema2 = pd.Series(ema1).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema3 = pd.Series(ema2).ewm(span=10, adjust=False, min_periods=10).mean().values
    trix = ema3
    
    # TRIX signal line: 9-period EMA of TRIX
    signal_line = pd.Series(trix).ewm(span=9, adjust=False, min_periods=9).mean().values
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_ma20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume_12h > (vol_ma20 * 1.5)
    
    # Align to 4h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_12h, trix)
    signal_line_aligned = align_htf_to_ltf(prices, df_12h, signal_line)
    vol_filter_aligned = align_htf_to_ltf(prices, df_12h, vol_filter)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(trix_aligned[i]) or np.isnan(signal_line_aligned[i]) or 
            np.isnan(vol_filter_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: TRIX crosses above signal line with volume filter
            long_cond = (trix_aligned[i] > signal_line_aligned[i] and 
                         trix_aligned[i-1] <= signal_line_aligned[i-1] and
                         vol_filter_aligned[i])
            
            # Short entry: TRIX crosses below signal line with volume filter
            short_cond = (trix_aligned[i] < signal_line_aligned[i] and 
                          trix_aligned[i-1] >= signal_line_aligned[i-1] and
                          vol_filter_aligned[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: TRIX crosses below signal line
            if trix_aligned[i] < signal_line_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: TRIX crosses above signal line
            if trix_aligned[i] > signal_line_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: TRIX (10,9) crossover with volume confirmation on 12h timeframe.
# TRIX filters out insignificant price movements and identifies momentum shifts.
# Volume filter ensures trades occur during periods of institutional participation.
# Works in both bull and bear markets by capturing momentum reversals.
# Target: 25-40 trades/year to minimize fee decay while capturing significant moves.