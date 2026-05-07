#!/usr/bin/env python3
"""
6h_RollingZScore_1wTrend_Volume
Hypothesis: Mean reversion at extreme Z-scores (2.0) with 1-week trend filter and volume confirmation captures reversals while avoiding trend-following whipsaws. The Z-score measures deviation from the 60-period mean, and extreme values indicate potential reversals. Weekly trend ensures alignment with higher timeframe momentum. Volume adds confirmation. Designed for low frequency and high win rate in both bull and bear markets.
Target: 50-150 total trades over 4 years.
"""
name = "6h_RollingZScore_1wTrend_Volume"
timeframe = "6h"
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
    
    # Z-score: (close - mean) / std over 60 periods
    period = 60
    mean = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    # Avoid division by zero
    z_score = np.where(std != 0, (close - mean) / std, 0.0)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, period - 1)  # Need enough data for Z-score
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(z_score[i]) or np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Z-score < -2.0 (oversold) + 1w uptrend + volume
            if z_score[i] < -2.0 and close[i] > ema_50_1w_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Z-score > 2.0 (overbought) + 1w downtrend + volume
            elif z_score[i] > 2.0 and close[i] < ema_50_1w_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: Z-score returns to zero (mean reversion)
            if position == 1:
                if z_score[i] >= 0.0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if z_score[i] <= 0.0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals