#!/usr/bin/env python3
# 12h_Trix_Trend_Volume
# Hypothesis: 12-hour TRIX crosses with daily EMA50 trend filter and volume confirmation.
# TRIX captures momentum shifts; EMA50 filters trend direction; volume confirms breakout strength.
# Designed for 12h to achieve 12-37 trades/year, suitable for both bull and bear markets.

name = "12h_Trix_Trend_Volume"
timeframe = "12h"
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
    
    # Daily data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Daily EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # TRIX on 12h (1-period EMA of 1-period EMA of 1-period EMA of roc)
    # ROC period = 12 (12-period rate of change)
    roc = np.full_like(close, np.nan)
    for i in range(12, n):
        if close[i-12] != 0:
            roc[i] = (close[i] - close[i-12]) / close[i-12]
    
    # Triple EMA of ROC
    ema1 = pd.Series(roc).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    trix = ema3  # TRIX is the third EMA of ROC
    
    # Daily volume confirmation: 20-period average
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p - 1, len(arr)):
                res[i] = np.mean(arr[i - p + 1:i + 1])
        return res
    vol_ma_20 = mean_arr(volume_1d, 20)
    
    # Align daily indicators to 12h timeframe (wait for 1d bar to close)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(trix[i]) or np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: TRIX crosses above zero, above daily EMA50, strong volume
            if trix[i] > 0 and trix[i-1] <= 0 and close[i] > ema_50_1d_aligned[i] and volume[i] > 2.0 * vol_ma_20_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below zero, below daily EMA50, strong volume
            elif trix[i] < 0 and trix[i-1] >= 0 and close[i] < ema_50_1d_aligned[i] and volume[i] > 2.0 * vol_ma_20_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: TRIX crosses below zero or below daily EMA50
            if trix[i] < 0 and trix[i-1] >= 0 or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: TRIX crosses above zero or above daily EMA50
            if trix[i] > 0 and trix[i-1] <= 0 or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals