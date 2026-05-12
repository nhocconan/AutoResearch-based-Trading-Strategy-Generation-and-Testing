#!/usr/bin/env python3
name = "4h_TRIX_Volume_Spike_Trend"
timeframe = "4h"
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
    
    # Load daily data for TRIX and volume average
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # TRIX: Triple EMA (15-period) of % change in EMA(15)
    # TRIX = EMA(EMA(EMA(close, 15), 15), 15) - then % change
    ema1 = pd.Series(close_1d).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    # Calculate % change of triple EMA
    trix_raw = np.zeros_like(ema3)
    trix_raw[1:] = (ema3[1:] - ema3[:-1]) / ema3[:-1] * 100
    # Signal line: EMA of TRIX (9-period)
    trix_signal = pd.Series(trix_raw).ewm(span=9, adjust=False, min_periods=9).mean().values
    # TRIX histogram: TRIX - signal line
    trix_hist = trix_raw - trix_signal
    trix_hist_aligned = align_htf_to_ltf(prices, df_1d, trix_hist)
    
    # Daily volume average (20-period) for volume spike filter
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # 4h EMA34 for trend filter
    ema_34 = pd.Series(close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(trix_hist_aligned[i]) or np.isnan(vol_ma_20_aligned[i]) or 
            np.isnan(ema_34[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: TRIX histogram crosses above zero + volume spike + above EMA34
            if (trix_hist_aligned[i] > 0 and trix_hist_aligned[i-1] <= 0 and 
                volume[i] > 1.5 * vol_ma_20_aligned[i] and close[i] > ema_34[i]):
                signals[i] = 0.25
                position = 1
            # Short: TRIX histogram crosses below zero + volume spike + below EMA34
            elif (trix_hist_aligned[i] < 0 and trix_hist_aligned[i-1] >= 0 and 
                  volume[i] > 1.5 * vol_ma_20_aligned[i] and close[i] < ema_34[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: TRIX histogram crosses below zero or price below EMA34
            if trix_hist_aligned[i] < 0 or close[i] < ema_34[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: TRIX histogram crosses above zero or price above EMA34
            if trix_hist_aligned[i] > 0 or close[i] > ema_34[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals