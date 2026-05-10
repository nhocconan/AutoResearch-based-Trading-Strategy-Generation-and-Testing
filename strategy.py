#!/usr/bin/env python3
# 4h_Trix_Momentum_Volume
# Hypothesis: TRIX momentum combined with volume spike and 1-day trend filter captures short-term momentum bursts in both bull and bear markets.
# Uses TRIX(12) for momentum, volume > 1.5x 20-period average for confirmation, and 1-day EMA50 for trend filter.
# Target: 25-35 trades/year to minimize fee drag while maintaining edge.

name = "4h_Trix_Momentum_Volume"
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
    
    # 1-day trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_1d_up = close_1d > ema50_1d
    trend_1d_down = close_1d < ema50_1d
    
    # Align 1-day trend to 4h
    trend_1d_up_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_up.astype(float))
    trend_1d_down_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_down.astype(float))
    
    # TRIX calculation: triple EMA of ROC
    # ROC = (close - close[n]) / close[n] * 100
    roc = np.zeros_like(close)
    for i in range(1, n):
        if close[i-1] != 0:
            roc[i] = (close[i] - close[i-1]) / close[i-1] * 100.0
        else:
            roc[i] = 0.0
    
    # Triple EMA of ROC
    ema1 = pd.Series(roc).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    trix = ema3  # TRIX is the triple EMA of ROC
    
    # Volume spike: > 1.5x 20-period average
    vol_ma = np.zeros_like(volume)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma[i] = vol_sum / 20
        else:
            vol_ma[i] = np.nan
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Need enough data for TRIX and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(trix[i]) or np.isnan(volume_spike[i]) or
            np.isnan(trend_1d_up_aligned[i]) or np.isnan(trend_1d_down_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: TRIX turning up (>0) with volume spike and uptrend
            if (trix[i] > 0 and trix[i-1] <= 0 and
                trend_1d_up_aligned[i] > 0.5 and
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: TRIX turning down (<0) with volume spike and downtrend
            elif (trix[i] < 0 and trix[i-1] >= 0 and
                  trend_1d_down_aligned[i] > 0.5 and
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: TRIX turns down or trend changes
            if (trix[i] < 0 and trix[i-1] >= 0) or \
               (trend_1d_up_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: TRIX turns up or trend changes
            if (trix[i] > 0 and trix[i-1] <= 0) or \
               (trend_1d_down_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals