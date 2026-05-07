#!/usr/bin/env python3
# 1D_Camarilla_R3S3_1WTrend_Volume_Signal
# Hypothesis: Weekly Camarilla R3/S3 breakout with 1w trend filter and volume confirmation on 1d timeframe.
# Uses weekly pivot levels for structure, 1w EMA for trend, and 1d volume spike for confirmation.
# Designed to work in both bull and bear markets by following higher timeframe trend.
# Target: 15-25 trades/year per symbol to stay well under limits.

name = "1D_Camarilla_R3S3_1WTrend_Volume_Signal"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for Camarilla pivot calculation and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    # Calculate Camarilla R3 and S3 levels from previous weekly period's OHLC
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Camarilla R3 and S3 levels
    hl_range = high_1w - low_1w
    r3_1w = close_1w + 1.1 * hl_range / 2
    s3_1w = close_1w - 1.1 * hl_range / 2
    
    # Align all levels to daily timeframe (use previous weekly period's levels)
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    
    # Calculate EMA34 for trend filter (weekly)
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume spike detection: 2.0x average volume (20-period for stability)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34)  # Ensure we have volume MA and EMA34 data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(r3_1w_aligned[i]) or np.isnan(s3_1w_aligned[i]) or 
            np.isnan(ema34_1w_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above weekly R3, price above weekly EMA34 (uptrend), volume spike (>2.0x)
            if (close[i] > r3_1w_aligned[i] and 
                close[i] > ema34_1w_aligned[i] and 
                volume[i] > 2.0 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly S3, price below weekly EMA34 (downtrend), volume spike (>2.0x)
            elif (close[i] < s3_1w_aligned[i] and 
                  close[i] < ema34_1w_aligned[i] and 
                  volume[i] > 2.0 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price returns to or below weekly S3 (opposite level)
            if close[i] <= s3_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price returns to or above weekly R3 (opposite level)
            if close[i] >= r3_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals