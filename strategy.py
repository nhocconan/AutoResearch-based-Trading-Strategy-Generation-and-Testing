#!/usr/bin/env python3
# 4h_TRIX_Volume_Spike_12hTrend
# Hypothesis: TRIX momentum combined with volume spikes and 12h EMA trend filter captures sustained moves in both bull and bear markets.
# Volume spikes confirm institutional interest, reducing false breakouts. Trend filter ensures alignment with higher timeframe momentum.
# Target: 20-30 trades/year with low turnover to minimize fee drag.

name = "4h_TRIX_Volume_Spike_12hTrend"
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
    volume = prices['volume'].values
    
    # TRIX (12-period EMA of EMA of EMA, then ROC)
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = ema1.ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = ema2.ewm(span=12, adjust=False, min_periods=12).mean()
    trix = 100 * (ema3.pct_change())
    trix_values = trix.values
    
    # Volume spike detection: current volume > 2.0 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # 12h EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure sufficient warmup for all indicators
    
    for i in range(start_idx, n):
        if np.isnan(trix_values[i]) or np.isnan(ema50_12h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: TRIX crosses above zero, volume spike, and price above 12h EMA50 (uptrend)
            if trix_values[i] > 0 and trix_values[i-1] <= 0 and volume_spike[i] and close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below zero, volume spike, and price below 12h EMA50 (downtrend)
            elif trix_values[i] < 0 and trix_values[i-1] >= 0 and volume_spike[i] and close[i] < ema50_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: TRIX crosses below zero or trend lost
            if trix_values[i] < 0 or close[i] < ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: TRIX crosses above zero or trend lost
            if trix_values[i] > 0 or close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals