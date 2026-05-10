#!/usr/bin/env python3
# 12h_TRIX_Trend_Filter
# Hypothesis: TRIX (Triple Exponential Average) captures momentum with reduced lag.
# In both bull and bear markets, TRIX crossovers above/below zero with volume confirmation
# and 1-week trend filter (price above/below 200-period EMA) capture sustained moves.
# Uses 12h timeframe for low trade frequency (target: 15-30/year) to minimize fee drag.

name = "12h_TRIX_Trend_Filter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-week data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    # 1-week EMA200 for trend (smooth, lag-appropriate)
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Calculate TRIX on 12h data
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    # TRIX: EMA(EMA(EMA(close, 15), 15), 15) - 1-period percent change
    ema1 = pd.Series(close_12h).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix_raw = (ema3[1:] - ema3[:-1]) / ema3[:-1] * 100  # percent change
    trix = np.concatenate([[np.nan], trix_raw])  # align with original length
    trix_aligned = align_htf_to_ltf(prices, df_12h, trix)
    
    # Volume confirmation (20-period average on 12h = ~10 days)
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p-1, len(arr)):
                res[i] = np.mean(arr[i-p+1:i+1])
        return res
    vol_ma = mean_arr(volume, 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(200, 20) + 5  # need enough history for calculations
    
    for i in range(start_idx, n):
        if np.isnan(trix_aligned[i]) or np.isnan(ema_200_1w_aligned[i]) or \
           np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirm = volume[i] > 1.5 * vol_ma[i] if vol_ma[i] > 0 else False
        
        if position == 0:
            # Long: TRIX crosses above zero with volume, above 1w EMA200 (uptrend)
            if trix_aligned[i] > 0 and trix_aligned[i-1] <= 0 and volume_confirm and close[i] > ema_200_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below zero with volume, below 1w EMA200 (downtrend)
            elif trix_aligned[i] < 0 and trix_aligned[i-1] >= 0 and volume_confirm and close[i] < ema_200_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: TRIX crosses below zero or price breaks below 1w EMA200
            if trix_aligned[i] < 0 or close[i] < ema_200_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: TRIX crosses above zero or price breaks above 1w EMA200
            if trix_aligned[i] > 0 or close[i] > ema_200_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals