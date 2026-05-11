#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeS
Hypothesis: Camarilla R3/S3 levels from daily data provide strong support/resistance.
Breakouts above R3 or below S3 with volume confirmation and daily trend filter capture
institutional breakouts. Works in bull (breakouts continue) and bear (breakdowns continue).
Target: 50-150 trades over 4 years (12-37/year) on 12h timeframe.
"""

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeS"
timeframe = "12h"
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
    
    # === DAILY DATA FOR CAMARILLA AND TREND ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels (R3, S3) from previous day
    # R3 = close + 1.1*(high - low)/2
    # S3 = close - 1.1*(high - low)/2
    camarilla_range = high_1d - low_1d
    r3_level = close_1d + 1.1 * camarilla_range / 2
    s3_level = close_1d - 1.1 * camarilla_range / 2
    
    # Use previous day's levels (avoid look-ahead)
    r3_prev = r3_level  # will be shifted by align function
    s3_prev = s3_level
    
    # Align Camarilla levels to 12h
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_prev)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_prev)
    
    # Daily trend filter: EMA34
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # === VOLUME CONFIRMATION (20-period average) ===
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or 
            np.isnan(ema34_aligned[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above R3 with volume > 1.5x average and price > daily EMA34 (uptrend)
            if (close[i] > r3_aligned[i] and 
                volume[i] > 1.5 * vol_ma20[i] and 
                close[i] > ema34_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below S3 with volume > 1.5x average and price < daily EMA34 (downtrend)
            elif (close[i] < s3_aligned[i] and 
                  volume[i] > 1.5 * vol_ma20[i] and 
                  close[i] < ema34_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price re-enters below R3 or volume drops significantly
            if close[i] < r3_aligned[i] or volume[i] < 0.5 * vol_ma20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price re-enters above S3 or volume drops significantly
            if close[i] > s3_aligned[i] or volume[i] < 0.5 * vol_ma20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals