#!/usr/bin/env python3
name = "12h_Adaptive_Trend_Follow"
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
    
    # Load 1w data for trend filter (primary)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA10 for trend filter
    ema_10_1w = pd.Series(close_1w).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema_10_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_10_1w)
    
    # Load 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d volume average for filter
    vol_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # Load 12h data for ATR
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate ATR(14) for 12h
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr1[0] = high_12h[0] - low_12h[0]
    tr2[0] = high_12h[0] - close_12h[0]
    tr3[0] = low_12h[0] - close_12h[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_aligned = align_htf_to_ltf(prices, df_12h, atr)
    
    # Calculate 12h EMA21 for entry timing
    ema_21_12h = pd.Series(close_12h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_21_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_10_1w_aligned[i]) or np.isnan(vol_avg_1d_aligned[i]) or 
            np.isnan(atr_aligned[i]) or np.isnan(ema_21_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price above 1w EMA10 AND above 12h EMA21 AND volume > 1.5x 1d avg
            if close[i] > ema_10_1w_aligned[i] and close[i] > ema_21_12h_aligned[i] and volume[i] > (1.5 * vol_avg_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price below 1w EMA10 AND below 12h EMA21 AND volume > 1.5x 1d avg
            elif close[i] < ema_10_1w_aligned[i] and close[i] < ema_21_12h_aligned[i] and volume[i] > (1.5 * vol_avg_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price below 1w EMA10 OR ATR-based stop
            if close[i] < ema_10_1w_aligned[i] or close[i] < (high[i] - 2.0 * atr_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price above 1w EMA10 OR ATR-based stop
            if close[i] > ema_10_1w_aligned[i] or close[i] > (low[i] + 2.0 * atr_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals