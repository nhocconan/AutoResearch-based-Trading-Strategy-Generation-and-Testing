#!/usr/bin/env python3
# 12h_Camarilla_R3_S3_1dTrend_Volume
# Hypothesis: 12h strategy using daily Camarilla R3/S3 levels with daily trend filter and volume confirmation.
# Enters long when price breaks above daily R3, close > daily EMA34 (uptrend), and volume > 2x average.
# Enters short when price breaks below daily S3, close < daily EMA34 (downtrend), and volume > 2x average.
# Exits when price returns to opposite S3/R3 level. Designed to avoid overtrading with strict entry conditions.
# Uses daily trend filter to work in both bull and bear markets by only trading in direction of higher timeframe trend.
# Target: 12h timeframe with daily HTF for trend and levels.

name = "12h_Camarilla_R3_S3_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous daily period's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate all Camarilla levels
    hl_range = high_1d - low_1d
    r3_1d = close_1d + 1.1 * hl_range / 2
    s3_1d = close_1d - 1.1 * hl_range / 2
    
    # Align all levels to 12h timeframe (use previous daily period's levels)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # Calculate EMA34 for trend filter (daily)
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume spike detection: 2.0x average volume (20-period for stability)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34)  # Ensure we have volume MA and EMA34 data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above daily R3, price above daily EMA34 (uptrend), volume spike (>2x)
            if (close[i] > r3_1d_aligned[i] and 
                close[i] > ema34_1d_aligned[i] and 
                volume[i] > 2.0 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below daily S3, price below daily EMA34 (downtrend), volume spike (>2x)
            elif (close[i] < s3_1d_aligned[i] and 
                  close[i] < ema34_1d_aligned[i] and 
                  volume[i] > 2.0 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price returns to or below daily S3 (opposite level)
            if close[i] <= s3_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price returns to or above daily R3 (opposite level)
            if close[i] >= r3_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals