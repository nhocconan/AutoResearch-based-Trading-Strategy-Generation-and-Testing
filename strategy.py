#!/usr/bin/env python3
"""
4h_RollingZScoreMeanReversion_1dTrend_Volume
Hypothesis: Mean reversion works in both bull and bear markets when filtered by daily trend and volume. 
Z-score of price relative to 100-period mean identifies overextended moves. 
Only trade when daily EMA50 confirms trend direction and volume confirms conviction.
Reduces false signals in choppy markets. Target: 50-150 trades over 4 years.
"""
name = "4h_RollingZScoreMeanReversion_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 120:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Z-score mean reversion: (price - mean) / std over 100 periods
    lookback = 100
    rolling_mean = pd.Series(close).rolling(window=lookback, min_periods=lookback).mean().values
    rolling_std = pd.Series(close).rolling(window=lookback, min_periods=lookback).std().values
    z_score = (close - rolling_mean) / rolling_std
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: current volume > 1.3 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(120, lookback)
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(z_score[i]) or np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price significantly below mean (oversold) + 1d uptrend + volume
            if z_score[i] < -1.8 and close[i] > ema_50_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price significantly above mean (overbought) + 1d downtrend + volume
            elif z_score[i] > 1.8 and close[i] < ema_50_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: price returns to mean (mean reversion complete) or adverse move
            if position == 1:
                if z_score[i] > -0.3 or close[i] < ema_50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if z_score[i] < 0.3 or close[i] > ema_50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals