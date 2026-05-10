#!/usr/bin/env python3
# 12H_1D_Camarilla_R3_S3_Breakout_Trend_Volume
# Hypothesis: Camarilla pivot levels from 1d act as strong support/resistance.
# Long when price breaks above R3 with volume confirmation and daily uptrend.
# Short when price breaks below S3 with volume confirmation and daily downtrend.
# Uses 1d EMA50 for trend filter and volume spike (volume > 1.5x average) for confirmation.
# Works in bull/bear by following daily trend direction. Target: 15-25 trades/year per symbol.

name = "12H_1D_Camarilla_R3_S3_Breakout_Trend_Volume"
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
    
    # Get daily data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for previous day
    # R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), etc.
    # S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
    camarilla_range = high_1d - low_1d
    r3 = close_1d + 1.1 * camarilla_range
    s3 = close_1d - 1.1 * camarilla_range
    
    # Daily EMA50 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume average for spike detection (20-period)
    volume_series = pd.Series(volume)
    vol_avg = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Align to 12h
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(ema50_aligned[i]) or np.isnan(vol_avg_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike condition: current volume > 1.5x average
        volume_spike = volume[i] > 1.5 * vol_avg_aligned[i]
        
        if position == 0:
            # Enter long: price breaks above R3, volume spike, daily uptrend
            if (high[i] > r3_aligned[i]) and volume_spike and (close[i] > ema50_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S3, volume spike, daily downtrend
            elif (low[i] < s3_aligned[i]) and volume_spike and (close[i] < ema50_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below S3 or daily downtrend
            if (low[i] < s3_aligned[i]) or (close[i] < ema50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above R3 or daily uptrend
            if (high[i] > r3_aligned[i]) or (close[i] > ema50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals