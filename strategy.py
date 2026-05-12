#!/usr/bin/env python3
# 12h_Camarilla_Pivot_R3_S3_Breakout_1dTrend_VolumeS
# Hypothesis: Combine daily Camarilla pivot levels (R3/S3) with 12h price breakouts and 1d trend filter.
# Long when price breaks above R3 with 1d uptrend and volume surge; short when breaks below S3 with 1d downtrend and volume surge.
# Exit on opposite break or trend failure. Uses volume confirmation to avoid false breakouts.
# Designed for low frequency (12-35 trades/year) to avoid fee drag. Works in bull/bear via trend filter.

name = "12h_Camarilla_Pivot_R3_S3_Breakout_1dTrend_VolumeS"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily EMA34 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate Camarilla levels from previous day
    # R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    camarilla_r3 = np.full(len(close_1d), np.nan)
    camarilla_s3 = np.full(len(close_1d), np.nan)
    
    for i in range(1, len(close_1d)):
        h = high_1d[i-1]
        l = low_1d[i-1]
        c = close_1d[i-1]
        camarilla_r3[i] = c + (h - l) * 1.1 / 2
        camarilla_s3[i] = c - (h - l) * 1.1 / 2
    
    # Align Camarilla levels and EMA to 12h timeframe
    r3_12h = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_12h = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    ema34_12h = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate volume average (20-period) for surge detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_surge = volume > (vol_ma * 2.0)  # Volume at least 2x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(r3_12h[i]) or np.isnan(s3_12h[i]) or 
            np.isnan(ema34_12h[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter: price vs EMA34 on 1d
        uptrend = close[i] > ema34_12h[i]
        downtrend = close[i] < ema34_12h[i]
        
        if position == 0:
            # LONG: Price breaks above R3 AND uptrend AND volume surge
            if close[i] > r3_12h[i] and uptrend and vol_surge[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 AND downtrend AND volume surge
            elif close[i] < s3_12h[i] and downtrend and vol_surge[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price breaks below S3 OR trend fails
            if close[i] < s3_12h[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R3 OR trend fails
            if close[i] > r3_12h[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals