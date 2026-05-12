#!/usr/bin/env python3
name = "4h_TRIX_VolumeSpike_12hTrend"
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
    
    # 12h trend filter: EMA50
    df_12h = get_htf_data(prices, '12h')
    ema50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # TRIX: triple EMA of ROC
    close_series = pd.Series(close)
    # ROC 1-period
    roc = close_series.pct_change(1)
    # EMA1 of ROC
    ema1 = roc.ewm(span=15, adjust=False, min_periods=15).mean().values
    # EMA2 of EMA1
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    # EMA3 of EMA2 (TRIX)
    trix = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    
    # Volume spike: volume > 1.5 * 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need enough data for TRIX and 12h EMA50
    
    for i in range(start_idx, n):
        # Skip if 12h trend data not ready
        if np.isnan(ema50_12h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: TRIX crosses above zero + volume spike + 12h uptrend
            if (trix[i] > 0 and trix[i-1] <= 0 and  # TRIX bullish crossover
                volume_spike[i] and                  # volume confirmation
                close[i] > ema50_12h_aligned[i]):    # 12h uptrend filter
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below zero + volume spike + 12h downtrend
            elif (trix[i] < 0 and trix[i-1] >= 0 and   # TRIX bearish crossover
                  volume_spike[i] and                   # volume confirmation
                  close[i] < ema50_12h_aligned[i]):     # 12h downtrend filter
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when TRIX crosses below zero
            if trix[i] < 0 and trix[i-1] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when TRIX crosses above zero
            if trix[i] > 0 and trix[i-1] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals