#!/usr/bin/env python3
# 1d_WH4_Signal_Momentum_1wTrend_Volume
# Hypothesis: Weekly Hull Moving Average (WH4) sets the trend direction on weekly timeframe.
# Daily price momentum (ROC10) confirms entry in direction of weekly trend with volume filter.
# WH4 is used for trend (less lag than SMA/EMA), ROC10 for momentum, volume for confirmation.
# Designed for low trade frequency (<15/year) to minimize fee drag and work in both bull/bear markets.
# Target: 10-15 trades/year on daily timeframe.

name = "1d_WH4_Signal_Momentum_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def whlma(arr, period):
    """Weighted Hull Moving Average: WMA(2*WMA(n/2) - WMA(n), sqrt(n))"""
    half = period // 2
    sqrt = int(np.sqrt(period))
    wma_half = pd.Series(arr).ewm(span=half, adjust=False).mean()
    wma_full = pd.Series(arr).ewm(span=period, adjust=False).mean()
    raw = 2 * wma_half - wma_full
    return pd.Series(raw).ewm(span=sqrt, adjust=False).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for trend (WH4)
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate weekly WH4 for trend
    wh4_1w = whlma(df_1w['close'].values, 4)
    wh4_1w_aligned = align_htf_to_ltf(prices, df_1w, wh4_1w)
    
    # Get daily data for signal
    close = prices['close'].values
    volume = prices['volume'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Daily ROC(10) for momentum
    roc10 = np.zeros_like(close)
    roc10[10:] = (close[10:] - close[:-10]) / close[:-10] * 100
    
    # Volume filter: current volume > 1.5x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ema20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need WH4 (4) + ROC10 (10) + vol EMA (20)
    start_idx = max(10, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(wh4_1w_aligned[i]) or 
            np.isnan(roc10[i]) or
            np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above WH4 (uptrend) + positive momentum + volume
            if close[i] > wh4_1w_aligned[i] and roc10[i] > 0 and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price below WH4 (downtrend) + negative momentum + volume
            elif close[i] < wh4_1w_aligned[i] and roc10[i] < 0 and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price below WH4 or momentum turns negative
            if close[i] < wh4_1w_aligned[i] or roc10[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price above WH4 or momentum turns positive
            if close[i] > wh4_1w_aligned[i] or roc10[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals