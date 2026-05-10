#!/usr/bin/env python3
# 4h_TRIX_ZeroCross_VolumeSpike_1dTrend
# Hypothesis: TRIX (triple smoothed EMA) captures momentum with reduced lag. 
# Zero-cross signals combined with 1d trend filter (EMA50) and volume spikes 
# produce high-quality entries in both bull and bear markets. 
# Volume confirmation reduces false signals. Target: 20-40 trades/year.

name = "4h_TRIX_ZeroCross_VolumeSpike_1dTrend"
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
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate TRIX on 4h data: triple EMA of ROC
    # ROC = (close / close.shift(1) - 1) * 100
    roc = np.zeros_like(close)
    roc[1:] = (close[1:] / close[:-1] - 1) * 100
    
    # Triple EMA: EMA(EMA(EMA(ROC)))
    ema1 = pd.Series(roc).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    trix = ema3  # TRIX is the final smoothed series
    
    # Volume confirmation (20-period MA on 4h = ~3.3 days)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need TRIX (12*3=36), 1d EMA50 (50), volume MA (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(trix[i]) or 
            np.isnan(trix[i-1]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # TRIX zero-cross signals
        trix_cross_up = trix[i-1] <= 0 and trix[i] > 0   # bullish momentum
        trix_cross_down = trix[i-1] >= 0 and trix[i] < 0  # bearish momentum
        
        # 1d trend filter
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        if position == 0:
            # Long entry: bullish TRIX cross + uptrend + volume
            if trix_cross_up and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: bearish TRIX cross + downtrend + volume
            elif trix_cross_down and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: bearish TRIX cross or trend breaks
            if trix_cross_down or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: bullish TRIX cross or trend breaks
            if trix_cross_up or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals