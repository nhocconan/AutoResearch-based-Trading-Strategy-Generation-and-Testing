#!/usr/bin/env python3
# 4h_TRIX_Signal_With_Volume_Confirm
# Hypothesis: TRIX (12,26,9) captures momentum reversals with less whipsaw than MACD.
# Long when TRIX crosses above zero with volume confirmation; short when crosses below zero with volume confirmation.
# Uses 1d EMA50 as trend filter to avoid counter-trend trades. Designed for low trade frequency (<40/year) to minimize fee drag.
# Works in bull markets (rides momentum) and bear markets (catches reversals) by filtering with 1d trend.

name = "4h_TRIX_Signal_With_Volume_Confirm"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate TRIX on 4h close: TRIX = EMA(EMA(EMA(close,12),12),12) then % change
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = ema1.ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = ema2.ewm(span=12, adjust=False, min_periods=12).mean()
    trix = 100 * (ema3 / ema3.shift(1) - 1)
    trix = trix.fillna(0).values
    
    # Signal line: 9-period EMA of TRIX
    signal_line = pd.Series(trix).ewm(span=9, adjust=False, min_periods=9).mean().values
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation (20-period MA on 4h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need TRIX calculation (12+12+12+1 = 37), signal line (9), 1d EMA50 (50), volume MA (20)
    start_idx = max(37, 9, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(trix[i]) or 
            np.isnan(signal_line[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # 1d trend filter
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        if position == 0:
            # Long entry: TRIX crosses above signal line + uptrend + volume
            if trix[i] > signal_line[i] and trix[i-1] <= signal_line[i-1] and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: TRIX crosses below signal line + downtrend + volume
            elif trix[i] < signal_line[i] and trix[i-1] >= signal_line[i-1] and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: TRIX crosses below signal line or trend breaks
            if trix[i] < signal_line[i] and trix[i-1] >= signal_line[i-1] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: TRIX crosses above signal line or trend breaks
            if trix[i] > signal_line[i] and trix[i-1] <= signal_line[i-1] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals