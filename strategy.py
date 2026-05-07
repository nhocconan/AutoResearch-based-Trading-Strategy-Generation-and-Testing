#!/usr/bin/env python3
# 12h_Camarilla_R3S3_Breakout_1dTrend_Volume_Spike
# Hypothesis: Camarilla pivot levels (R3/S3) on 12h combined with 1-day EMA34 trend filter and volume spike captures institutional breakouts. Works in bull/bear by trading breakouts in direction of higher timeframe trend. Target: 15-30 trades/year per symbol.

name = "12h_Camarilla_R3S3_Breakout_1dTrend_Volume_Spike"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Camarilla levels for each 12h bar using previous 1d OHLC
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), S3 = close - 1.1*(high-low)
    cam_r3 = close_1d + 1.1 * (high_1d - low_1d)
    cam_s3 = close_1d - 1.1 * (high_1d - low_1d)
    cam_r3_aligned = align_htf_to_ltf(prices, df_1d, cam_r3)
    cam_s3_aligned = align_htf_to_ltf(prices, df_1d, cam_s3)
    
    # Volume spike detection: 2.5x average volume (60-period = ~5 days on 12h chart)
    vol_ma = pd.Series(volume).rolling(window=60, min_periods=60).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 60)  # Ensure we have EMA34 and volume MA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0 or
            np.isnan(cam_r3_aligned[i]) or np.isnan(cam_s3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Camarilla R3, price above EMA34 (uptrend), volume spike
            if (high[i] > cam_r3_aligned[i-1] and 
                close[i] > ema34_1d_aligned[i] and 
                volume[i] > 2.5 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S3, price below EMA34 (downtrend), volume spike
            elif (low[i] < cam_s3_aligned[i-1] and 
                  close[i] < ema34_1d_aligned[i] and 
                  volume[i] > 2.5 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price breaks below Camarilla S3 OR price crosses below EMA34
            if (low[i] < cam_s3_aligned[i-1] or close[i] < ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price breaks above Camarilla R3 OR price crosses above EMA34
            if (high[i] > cam_r3_aligned[i-1] or close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals