#!/usr/bin/env python3
# 4h_TRIX_Zero_Cross_Volume_Filter
# Hypothesis: TRIX (triple exponential average) zero cross with volume confirmation and 1d EMA trend filter.
# TRIX captures momentum shifts; zero cross indicates change in momentum direction.
# Volume ensures breakouts have conviction. 1d EMA filter aligns with higher timeframe trend.
# Works in bull/bear by trading momentum shifts with trend alignment.
# Targets 20-30 trades/year to minimize fee drag.

name = "4h_TRIX_Zero_Cross_Volume_Filter"
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
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 trend
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_1d_up = close_1d > ema34_1d
    trend_1d_down = close_1d < ema34_1d
    
    # Align 1d trend to 4h
    trend_1d_up_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_up.astype(float))
    trend_1d_down_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_down.astype(float))
    
    # TRIX calculation (15-period standard)
    # TRIX = EMA(EMA(EMA(close, 15), 15), 15)
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    
    # TRIX = (ema3 - previous ema3) / previous ema3 * 100
    trix = np.zeros_like(close)
    trix[1:] = (ema3[1:] - ema3[:-1]) / ema3[:-1] * 100
    # First value remains 0 (no previous value)
    
    # Volume filter: current volume > 1.5 * 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(trix[i]) or np.isnan(trend_1d_up_aligned[i]) or 
            np.isnan(trend_1d_down_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_filter = vol_ratio > 1.5
        
        if position == 0:
            # Long: TRIX crosses above zero with uptrend and volume
            if (trix[i] > 0 and trix[i-1] <= 0 and
                trend_1d_up_aligned[i] > 0.5 and
                volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below zero with downtrend and volume
            elif (trix[i] < 0 and trix[i-1] >= 0 and
                  trend_1d_down_aligned[i] > 0.5 and
                  volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: TRIX crosses below zero or trend fails
            if (trix[i] < 0 and trix[i-1] >= 0) or \
               trend_1d_up_aligned[i] < 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: TRIX crosses above zero or trend fails
            if (trix[i] > 0 and trix[i-1] <= 0) or \
               trend_1d_down_aligned[i] < 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals